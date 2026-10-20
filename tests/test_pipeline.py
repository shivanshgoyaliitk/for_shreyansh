import ast
import json, shutil
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import preprocessing as pp
from validation import assign_folds, macro_auc, rank_blend
from model_core import KneeNet, SlotQueryNet, Reader25D, RadImageNetHead, build_model, slice_thirds, spread_groups, window_centres
import label_builder as lb
import copy

ROOT = Path(__file__).resolve().parents[1]

def cell(file, index): return ''.join(json.loads((ROOT/file).read_text())['cells'][index]['source'])


def write_atomic(path, content):
    temp = f'{path}.{os.getpid()}.tmp'
    if callable(content): content(temp)
    else: Path(temp).write_text(content)
    os.replace(temp, path)



def test_notebook_syntax():
    from IPython.core.inputtransformer2 import TransformerManager
    transform = TransformerManager()
    for file in ['training.ipynb','submission-notebook.ipynb','knee_mri_step2c_cache.ipynb']:
        for c in json.loads((ROOT/file).read_text())['cells']:
            if c['cell_type']=='code': ast.parse(transform.transform_cell(''.join(c['source'])))


def test_center_geometry_and_tag_priority():
    # Corner is on the right; image center is on the left. Unequal row/column spacings matter.
    row = dict(ImagePositionPatient='-50|0|0', ImageOrientationPatient='1|0|0|0|1|0',
               PixelSpacing='1|2',Rows=100,Columns=100)
    assert pp.series_side(row)==('L','coords')
    row['ImageLaterality']='R'
    assert pp.series_side(row)==('R','tag')
    group=pd.DataFrame(dict(side=['R','L','L'],side_src=['tag','coords','coords']))
    assert pp.vote(group)==('R','tag')
    assert pp.vote(pd.DataFrame(dict(side=['L','R'],side_src=['tag','tag'])))[1]=='default'
    assert pp.series_side({})==(None,'unknown')


def test_direction_is_scanner_independent():
    a=SimpleNamespace(ImageOrientationPatient=[1,0,0,0,1,0], ImagePositionPatient=[0,0,8])
    b=SimpleNamespace(ImageOrientationPatient=[-1,0,0,0,1,0], ImagePositionPatient=[0,0,8])
    assert pp._sort_key(a)==pp._sort_key(b)==8


def fake_series(monkeypatch, tmp_path, failed=()):
    for i in range(5): (tmp_path/f'{i}.dcm').touch()
    def read(path,stop_before_pixels=False):
        i=int(Path(path).stem)
        if i in failed and not stop_before_pixels: raise ValueError('corrupt pixels')
        return SimpleNamespace(ImageOrientationPatient=[1,0,0,0,0,-1],
               ImagePositionPatient=[0,i,0],PixelSpacing=[1,2],
               pixel_array=(np.arange(80*80).reshape(80,80)+i*100).astype(np.float32))
    monkeypatch.setattr(pp.pydicom,'dcmread',read)


def test_decode_failed_first_slice_and_laterality(monkeypatch,tmp_path):
    fake_series(monkeypatch,tmp_path,failed=(0,))
    left=pp.load_series(str(tmp_path),'Coronal','L',n_slices=5,img=16,band=(0,1),crop_mm=40)
    right=pp.load_series(str(tmp_path),'Coronal','R',n_slices=5,img=16,band=(0,1),crop_mm=40)
    assert left.dtype==np.uint8 and left.shape==(5,16,16)
    np.testing.assert_array_equal(left[0],left[1])
    np.testing.assert_array_equal(left[:,:,::-1],right)


def test_anisotropic_crop(monkeypatch,tmp_path):
    fake_series(monkeypatch,tmp_path)
    sizes=[]
    resize=pp.cv2.resize
    def capture(a,*args,**kwargs): sizes.append(a.shape); return resize(a,*args,**kwargs)
    monkeypatch.setattr(pp.cv2,'resize',capture)
    pp.load_series(str(tmp_path),'Coronal','L',n_slices=2,img=16,crop_mm=40)
    assert sizes==[(40,20),(40,20)]


def test_empty_manifest():
    studies=pd.DataFrame({'StudyInstanceUID':['missing']})
    # Empty tables still must yield a row for sample-submission alignment.
    series=pd.DataFrame(columns=['StudyInstanceUID','series_dir','Fluid_Sensitive','Anatomical_Plane'])
    out=pp.build_manifest(studies,series)
    assert out.loc[0,'dir_0']==''


def test_folds_gold_and_group_isolation():
    rng=np.random.default_rng(4)
    data=pd.DataFrame(rng.integers(0,2,(40,3)),columns=['a','b','c'])
    data['StudyInstanceUID']=[f'id{i}' for i in range(40)]
    data['source']=['pseudo']*36+['gold']*4
    data['patient']=[i//2 for i in range(40)]
    folds=assign_folds(data,['a','b','c'],3,42,'patient')
    assert (folds[-4:]==-1).all()
    assert all(len(set(folds[i:i+2]))==1 for i in range(0,36,2))
    np.testing.assert_array_equal(folds,assign_folds(data,['a','b','c'],3,42,'patient'))
    plain=assign_folds(data,['a','b','c'],3,42)
    assert set(plain)=={-1,0,1,2}


def test_masked_loss_and_rank():
    import torch.nn.functional as F
    namespace={'torch':torch,'F':F}
    exec(cell('training.ipynb',13),namespace)
    logits=torch.zeros(2,2,requires_grad=True)
    loss=namespace['masked_bce'](logits,torch.tensor([[1.,float('nan')],[0.,.5]]))
    loss.backward(); assert logits.grad[0,1]==0
    p=np.array([[.2,.9],[.1,.3],[.8,.4]])
    np.testing.assert_allclose(rank_blend([p,p**2]),rank_blend([p]))
    auc,_=macro_auc(np.array([[0.,np.nan],[1.,np.nan]]),np.array([[0.,0.],[1.,0.]]),['a','b'])
    assert auc==1.


class TinyEncoder(nn.Module):
    num_features=4
    pretrained_cfg={'input_size':(3,32,32),'mean':(.485,.456,.406),'std':(.229,.224,.225)}
    def __init__(self): super().__init__(); self.conv=nn.Conv2d(3,4,1)
    def forward(self,x): return self.conv(x).mean((-2,-1))


def patch_encoder(monkeypatch):
    import timm
    monkeypatch.setattr(timm,'create_model',lambda *a,**k:TinyEncoder())


def test_legacy_checkpoint_predictions_unchanged(monkeypatch):
    patch_encoder(monkeypatch)
    namespace=dict(torch=torch,nn=nn,BACKBONE='tiny',LABEL_COLS=['a','b'],PLANES=['x','y'])
    code=cell('backups/original/submission-notebook.ipynb',8).split('# pretrained=False')[0]
    exec(code,namespace)
    old=namespace['KneeNet']('tiny',2,2).eval()
    new=KneeNet('tiny',2,2,pretrained=False,encoder_chunk=3).eval()
    new.load_state_dict(old.state_dict(),strict=True)
    x=torch.randint(0,256,(2,2,4,16,16),dtype=torch.uint8)
    present=torch.tensor([[True,False],[True,True]])
    with torch.no_grad():
        expected=old(x,present)[0]; actual=new(x,present)[0]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_real_backbones_forward():
    # Architecture/shape check uses random weights; no external weight download.
    for name in ['tf_efficientnet_b0.ns_jft_in1k','vit_small_patch16_dinov3.lvd1689m']:
        model=KneeNet(name,2,1,pretrained=False,img=32,normalization='pretrained').eval()
        with torch.inference_mode():
            out,att=model(torch.zeros((1,1,2,32,32),dtype=torch.uint8),torch.ones((1,1),dtype=torch.bool))
        assert out.shape==(1,2) and torch.isfinite(out).all()


def test_cache_resume_and_recipe_guard(monkeypatch,tmp_path):
    monkeypatch.setattr(pp,'load_series',lambda *a,**k:np.ones((2,8,8),np.uint8))
    import multiprocessing as mp
    monkeypatch.setattr(mp,'get_all_start_methods',lambda:[])
    meta=dict(pp.preprocessing_meta(),img=8,n_slices=2)
    manifest=pd.DataFrame([dict(StudyInstanceUID='x',study_side='L',side_src='tag',dir_0='a',dir_1='',dir_2='b',
                                dir_3='',dir_4='c',dir_5='',dir_6='a',dir_7='')])
    from tqdm.auto import tqdm
    namespace=dict(vars(pp),manifest=manifest,WORK=str(tmp_path),CACHE=str(tmp_path/'cache.npy'),
                   INDEX=str(tmp_path/'cache_index.csv'),METAJ=str(tmp_path/'cache_meta.json'),
                   IMG=8,N_SLICES=2,tqdm=tqdm,preprocessing_meta=lambda:meta)
    code=(ROOT/'cache_runtime.py').read_text()
    exec(code,namespace)
    first=namespace['present'].copy()
    exec(code,namespace)
    np.testing.assert_array_equal(first,namespace['present'])
    assert first.tolist()==[[True,False,True,False,True,False,True,False]]
    namespace['manifest']=manifest.assign(study_side='R')
    with pytest.raises(AssertionError,match='recipe'):exec(code,namespace)


def test_training_and_submission_smoke(monkeypatch,tmp_path):
    import gc, random,time,glob,warnings,traceback,cv2
    from torch.utils.data import Dataset,DataLoader
    from tqdm.auto import tqdm
    from validation import assign_folds,macro_auc,rank_blend
    patch_encoder(monkeypatch)
    n=16; labels=['a','b']; cache_dir=tmp_path/'cache'; cache_dir.mkdir()
    rng=np.random.default_rng(5)
    np.save(cache_dir/'cache.npy',rng.integers(0,256,(n,3,2,8,8),dtype=np.uint8))
    df=pd.DataFrame(dict(StudyInstanceUID=[f'id{i}' for i in range(n)],source=['pseudo']*12+['gold']*4,
                         row=range(n),a=[0,1]*8,b=[1,0,0,1]*4,**{f'present_{i}':[True]*n for i in range(3)}))
    out=tmp_path/'out'; out.mkdir()
    namespace=dict(os=os,gc=gc,math=__import__('math'),time=time,json=json,random=random,np=np,pd=pd,
                   torch=torch,nn=nn,F=torch.nn.functional,Dataset=Dataset,DataLoader=DataLoader,tqdm=tqdm,
                   df=df,CACHE_DIR=str(cache_dir),CACHE_META=pp.preprocessing_meta('knee-2c-v2'),PLANES=['SAG_FLUID','COR_FLUID','AX_FLUID'],
                   ARCH='kneenet',UNFREEZE_LAST=None,HEAD_HIDDEN=None,PCT_START=.1,build_model=build_model,slice_thirds=slice_thirds,
                   copy=copy,CACHE_DIRS=[str(cache_dir)],GOLD_WEIGHT=0.,EMA_DECAY=0.,CLIP=5.,TRAIN_WINDOWS=6,
                   PRESET='kneenet_b0',GRAD_CHECKPOINT=False,SLOT_DROPOUT=0.0,TEACHER_SUMMARY=None,write_atomic=write_atomic,
                   LABEL_COLS=labels,T_TRAIN=2,BATCH=2,NUM_WORKERS=0,DEV='cpu',OUT=str(out),
                   N_FOLDS=2,SEED=42,GROUP_COL=None,BACKBONE='tiny',IMG=8,N_SLICES=2,
                   NORMALIZATION='pretrained',DROPOUT=.4,EPOCHS=1,LR=3e-4,WD=.01,ENCODER_LR_SCALE=.3,
                   FOLDS_TO_RUN=[0,1],ACCUM_STEPS=2,RESUME=False,assign_folds=assign_folds,macro_auc=macro_auc,rank_blend=rank_blend,KneeNet=KneeNet)
    exec(cell('training.ipynb',9),namespace)
    exec(cell('training.ipynb',13),namespace)
    exec((ROOT/'train_runtime.py').read_text(),namespace)
    assert len(pd.read_csv(out/'oof_predictions.csv'))==12
    assert (out/'fold_0.pt').exists() and (out/'fold_1.pt').exists()
    namespace['RESUME']=True
    exec((ROOT/'train_runtime.py').read_text(),namespace)
    # Submit two known-good fake decodes and one entirely blank study.
    comp=tmp_path/'competition';comp.mkdir()
    sample=pd.DataFrame(dict(StudyInstanceUID=['s2','s0','s1'],a=[0]*3,b=[0]*3))
    sample.to_csv(comp/'sample_submission.csv',index=False)
    manifest=pd.DataFrame(dict(StudyInstanceUID=['s0','s1','s2'],study_side=['L']*3,
                        dir_0=['one','two',''],dir_1=['','',''],dir_2=['','','']))
    cfg=torch.load(out/'fold_0.pt',weights_only=False)['cfg']
    namespace.update(glob=glob,traceback=traceback,manifest=manifest,MODERN=True,SLOTS=pp.RECIPES['knee-2c-v2']['slots'],cfg=cfg,SERIES_KW={},
                    RECIPE=pp.RECIPES['knee-2c-v2'],slot_kwargs=pp.slot_kwargs,row_anchor=pp.row_anchor,
                    CKPT_PATHS=[str(out/f'fold_{i}.pt') for i in range(2)],COMP_DIR=str(comp),
                    OUT_CSV=str(out/'submission.csv'),MAX_HOURS=1,rank_blend=rank_blend,
                    load_series=lambda *a,**k:rng.integers(0,256,(2,8,8),dtype=np.uint8))
    exec((ROOT/'inference_runtime.py').read_text(),namespace)
    submitted=pd.read_csv(out/'submission.csv')
    assert submitted.StudyInstanceUID.tolist()==sample.StudyInstanceUID.tolist()
    assert submitted[labels].notna().all().all()
    diagnostics=json.loads((out/'inference_diagnostics.json').read_text())
    assert diagnostics['blank_studies']==1 and diagnostics['ensemble_members']==2


def test_launcher_input_discovery(tmp_path):
    from kaggle_inputs import one_input, fold_inputs, attached_files
    mount = tmp_path / 'notebooks' / 'owner' / 'cache'
    mount.mkdir(parents=True)
    (mount / 'cache_meta.json').write_text('{}')
    wheelhouse = mount / 'wheelhouse'
    wheelhouse.mkdir()
    (wheelhouse / 'timm-1.0.29-py3-none-any.whl').touch()
    assert one_input('cache_meta.json', root=tmp_path) == str(mount / 'cache_meta.json')
    assert len(attached_files('timm-*.whl', root=tmp_path)) == 1
    for fold in range(5):
        folder = tmp_path / f'train-{fold}'
        folder.mkdir()
        (folder / f'fold_{fold}.pt').touch()
    assert len(fold_inputs(range(5), root=tmp_path)) == 5
    with pytest.raises(RuntimeError, match='exactly one'):
        one_input('train_targets.csv', root=tmp_path)
    duplicate = tmp_path / 'duplicate'
    duplicate.mkdir()
    (duplicate / 'fold_0.pt').touch()
    with pytest.raises(RuntimeError, match='exactly one'):
        fold_inputs(range(5), root=tmp_path)
    # The DICOM trees are never listed, and files deeper than five folders are not artifacts.
    dicom = tmp_path / 'comp' / 'train_series' / 'study'
    dicom.mkdir(parents=True)
    (dicom / 'cache_meta.json').write_text('{}')
    deep = tmp_path / 'a' / 'b' / 'c' / 'd' / 'e' / 'f'
    deep.mkdir(parents=True)
    (deep / 'cache_meta.json').write_text('{}')
    (deep.parent / 'cache_meta.json').write_text('{}')
    assert attached_files('cache_meta.json', root=tmp_path) == sorted([mount / 'cache_meta.json',
                                                                      deep.parent / 'cache_meta.json'])


def test_launch_notebooks_clean_and_syntax():
    from IPython.core.inputtransformer2 import TransformerManager
    transform = TransformerManager()
    notebooks = sorted((ROOT / 'RUN_ON_KAGGLE').glob('*.ipynb'))
    assert len(notebooks) == 86
    for path in notebooks:
        notebook = json.loads(path.read_text())
        import nbformat
        nbformat.validate(notebook)
        assert 'widgets' not in notebook['metadata']
        for c in notebook['cells']:
            if c['cell_type'] == 'code':
                assert c['outputs'] == [] and c['execution_count'] is None
                ast.parse(transform.transform_cell(''.join(c['source'])))
    for fold in range(5):
        primary = cell(f'RUN_ON_KAGGLE/04_train_fold{fold}.ipynb', 5)
        second = cell(f'RUN_ON_KAGGLE/04b_train_slot_fold{fold}.ipynb', 5)
        assert f'FOLDS_TO_RUN = [{fold}]' in primary and f'FOLDS_TO_RUN = [{fold}]' in second
        assert 'PRESET = "reader_convnext"' in primary and 'PRESET = "slot_dinov2"' in second
    rad = json.loads((ROOT/'RUN_ON_KAGGLE/04d_train_radimagenet.ipynb').read_text())['cells']
    assert 'PRESET = "radimagenet"' in ''.join(rad[5]['source']) and 'FOLDS_TO_RUN = [0, 1, 2, 3, 4]' in ''.join(rad[5]['source'])
    assert 'RadImageNet head family' in ''.join(rad[16]['source'])
    for shard in range(2):
        code = ''.join(''.join(c['source']) for c in json.loads((ROOT/f'RUN_ON_KAGGLE/02_cache_shard{shard}.ipynb').read_text())['cells']
                       if c['cell_type'] == 'code')
        assert f'SHARD = {shard}' in code and ('pip\', \'download' in code or 'pip download' in code or "'download'" in code) == (shard == 0)
    assert 'CKPT_PATHS = checkpoint_inputs()' in cell('RUN_ON_KAGGLE/05_submit.ipynb', 1)
    assert not list((ROOT / 'RUN_ON_KAGGLE').glob('04c_*'))            # CoAtNet @384 does not fit a T4
    jobs = set()
    for name in ['04_session_A_reader_f0_f1', '04_session_B_reader_f2_f3', '04_session_C_reader_f4_slot_rad']:
        ns = {}
        exec(cell(f'RUN_ON_KAGGLE/{name}.ipynb', 2), ns)
        jobs |= {(preset, f) for queue in ns['GPU_JOBS'].values() for preset, folds in queue for f in folds}
        exec(cell(f'RUN_ON_KAGGLE/{name}.ipynb', 3), ns)
        assert 'KNEE_FOLDS' in ns['TRAIN_JOB_SOURCE'] and 'Five-fold training runtime' in ns['TRAIN_JOB_SOURCE']
        assert 'RadImageNet head family' in ns['RAD_JOB_SOURCE'] and 'Five-fold training runtime' not in ns['RAD_JOB_SOURCE']
    # The three sessions cover every fold of every family exactly once.
    assert jobs == {(p, f) for p in ('reader_convnext', 'slot_dinov2', 'radimagenet') for f in range(5)}
    # The extra reader seeds (sessions D, E, F, H, I) cover seeds 7 and 11 fold by fold, once each.
    seeded = []
    for name in ['04_session_D_reader_s7_f0_f1', '04_session_E_reader_s7_f2_f3', '04_session_F_reader_s7_f4_s11_f0',
                 '04_session_H_reader_s11_f1_f2', '04_session_I_reader_s11_f3_f4']:
        ns = {}
        exec(cell(f'RUN_ON_KAGGLE/{name}.ipynb', 2), ns)
        seeded += [(preset, seed, f) for queue in ns['GPU_JOBS'].values() for preset, folds, seed in queue for f in folds]
    assert sorted(seeded) == sorted(('reader_convnext', s, f) for s in (7, 11) for f in range(5))


def test_v2_metadata_unchanged():
    # v2 checkpoints validate against this exact dictionary at inference.
    assert pp.preprocessing_meta('knee-2c-v2') == dict(
        version='knee-2c-v2', slots=['SAG_FLUID','COR_FLUID','AX_FLUID'],
        planes=['Sagittal','Coronal','Axial'], n_slices=12, img=336, crop_mm=130.0,
        slice_band=[0.2,0.8], canon_side='L', min_slices=8)
    v3 = pp.preprocessing_meta('knee-2c-v3')
    assert len(v3['slots']) == 6 and 'local' not in v3          # v3 checkpoints still validate
    assert (v3['n_slices'], v3['windows'], v3['img']) == (24, 8, 240)
    meta = pp.preprocessing_meta('knee-2c-v4')
    assert meta['slots'][:6] == v3['slots'] and len(meta['slots']) == 8
    assert meta['local'] == {'SAG_LOCAL': ['SAG_FLUID_FS', 'SAG_FLUID_NOFS'], 'COR_LOCAL': ['COR_FLUID_FS', 'COR_T1']}
    v5 = pp.preprocessing_meta()
    assert v5['version'] == 'knee-2c-v5' and v5['slots'] == meta['slots'] and v5['local'] == meta['local']
    assert (v5['img'], v5['n_slices'], v5['windows']) == (288, 30, 10)
    assert pp.series_kwargs(pp.recipe('knee-2c-v2'))['n_slices'] == 12
    assert pp.series_kwargs(pp.recipe('knee-2c-v2'))['windows'] is None


def test_v4_joint_anchor_and_local_crop(monkeypatch, tmp_path):
    # Joint centre: LR from the sagittal stack, AP from the coronal stack, SI from both.
    np.testing.assert_allclose(pp.joint_anchor(np.array([10., 5, 3]), np.array([-2., 40, 7])), [10, 40, 5])
    assert pp.joint_anchor(None, None) is None
    ds = SimpleNamespace(ImagePositionPatient=[0, 0, 0], ImageOrientationPatient=[1, 0, 0, 0, 0, -1], PixelSpacing=[1, 2])
    assert pp.project_anchor(ds, [40, 0, -30], (80, 80), False) == (30, 20)
    assert pp.project_anchor(ds, [40, 0, -30], (80, 80), True) == (30, 59)
    assert pp.project_anchor(ds, [20, 0, -30], (80, 80), False) is None      # column 10 < 15% of 80
    out = pp.square_crop(np.ones((10, 10)), 1, 1, 3, 3)
    assert out.shape == (6, 6) and out[:2].sum() == 0 and out[2:, 2:].all()
    # Local slots: 90 mm around the anchor; regular slots and v3 keep the centred 130 mm crop.
    v4, v3 = pp.recipe('knee-2c-v4'), pp.recipe('knee-2c-v3')
    kw = pp.slot_kwargs(v4, 6, [1., 2, 3])
    assert kw['crop_mm'] == 90.0 and list(kw['center_lps']) == [1, 2, 3]
    assert 'center_lps' not in pp.slot_kwargs(v4, 6, [np.nan, 2, 3]) and pp.slot_kwargs(v4, 0, [1, 2, 3]) == pp.series_kwargs(v4)
    assert pp.slot_kwargs(v3, 0) == pp.series_kwargs(v3)
    assert pp.row_anchor({'anchor_x': 1, 'anchor_y': np.nan, 'anchor_z': 0}) is None and pp.row_anchor({}) is None
    # The crop follows the projected anchor (fake coronal series: value = 80*row + col + 100*slice).
    fake_series(monkeypatch, tmp_path)
    crops = []
    resize = pp.cv2.resize
    monkeypatch.setattr(pp.cv2, 'resize', lambda a, *x, **k: (crops.append(a.copy()), resize(a, *x, **k))[1])
    pp.load_series(str(tmp_path), 'Coronal', 'L', n_slices=1, img=16, band=(0, 0), crop_mm=40, center_lps=[60, 0, -40])
    pp.load_series(str(tmp_path), 'Coronal', 'L', n_slices=1, img=16, band=(0, 0), crop_mm=40)
    (anchored,), (centred,) = crops[:1], crops[1:]
    assert anchored.shape == centred.shape == (40, 20)
    assert anchored[0, 0] == 80 * 20 + 20 + 200 and centred[0, 0] == 80 * 20 + 30 + 200   # middle slice: +100*2


def test_v4_local_slots_reuse_source_series():
    group = pd.DataFrame(dict(plane=['Sagittal', 'Coronal', 'Coronal'], fluid=[True, True, False],
                              fatsat=[False, True, False], weight=['PD', 'PD', 'T1'], n_slices=[20, 20, 20],
                              series_dir=['sag_nofs', 'cor_fs', 'cor_t1']))
    picked = pp.choose(group, pp.recipe('knee-2c-v4'))
    assert picked['SAG_LOCAL']['series_dir'] == 'sag_nofs' and picked['COR_LOCAL']['series_dir'] == 'cor_fs'
    assert 'SAG_FLUID_FS' not in picked and picked['COR_T1']['series_dir'] == 'cor_t1'


def test_window_slices_are_adjacent():
    idx = pp.slice_indices(40, 24, (0.06, 0.94), 8).reshape(8, 3)
    assert (np.diff(idx, axis=1) == 1).all()                  # each window: three neighbours
    assert idx[0, 1] <= 3 and idx[-1, 1] >= 36                # centres span the stack
    assert (np.diff(idx[:, 1]) > 0).all()
    short = pp.slice_indices(2, 24, (0.06, 0.94), 8)
    assert short.min() >= 0 and short.max() <= 1 and len(short) == 24


def test_v3_fatsat_typing():
    df = pd.DataFrame(dict(
        SeriesDescription=['PD FS SAG', 'T2_SPAIR', 'STIR cor', 'PD fat sat', 'T1 SE', 'PD tse', 'pd'],
        SequenceName=['']*7, ScanOptions=['', '', '', '', 'SP', '', 'FS'], ScanningSequence=['SE']*7,
        RepetitionTime=[3000]*4+[500]+[3000]*2, EchoTime=[30]*7))
    out = pp.annotate(df)
    assert out['fatsat'].tolist() == [True, True, True, True, False, False, True]
    assert out['fatsat_hdr'].tolist()[4]          # the v2 typing still reads SP as fat-sat


def series_group(rows):
    cols = ['plane', 'fluid', 'fatsat', 'weight', 'n_slices', 'series_dir']
    return pd.DataFrame(rows, columns=cols)


def test_v3_strict_slots():
    group = series_group([
        ('Sagittal', True, True, 'PD', 30, 'sag_pd_fs'),
        ('Sagittal', True, False, 'PD', 32, 'sag_pd'),
        ('Sagittal', False, False, 'GRE', 40, 'sag_gre'),
        ('Sagittal', False, False, 'T1', 20, 'sag_t1'),
        ('Coronal', True, False, 'PD', 24, 'cor_pd'),       # no fat-suppressed coronal fluid
        ('Axial', True, True, 'T2', 3, 'ax_short'),
        ('Axial', True, True, 'T2', 18, 'ax_t2_fs'),
    ])
    picked = {k: v['series_dir'] for k, v in pp.choose(group, pp.recipe('knee-2c-v3')).items()}
    assert picked == {'SAG_FLUID_FS': 'sag_pd_fs', 'SAG_FLUID_NOFS': 'sag_pd',
                      'SAG_T1': 'sag_t1', 'AX_FLUID_FS': 'ax_t2_fs'}
    # v2 keeps its fluid-with-fallback selection.
    v2 = {k: v['series_dir'] for k, v in pp.choose(group, pp.recipe('knee-2c-v2')).items()}
    assert v2 == {'SAG_FLUID': 'sag_pd', 'COR_FLUID': 'cor_pd', 'AX_FLUID': 'ax_t2_fs'}


class TinyViT(nn.Module):
    # Token encoder with the timm attributes SlotQueryNet relies on.
    num_features = 4
    num_prefix_tokens = 1
    pretrained_cfg = {'input_size': (3, 32, 32), 'mean': (.485, .456, .406), 'std': (.229, .224, .225)}
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 4, 4, 4)
        self.cls = nn.Parameter(torch.zeros(1, 1, 4))
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.norm = nn.LayerNorm(4)
    def forward_features(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        for block in self.blocks: t = t + block(t)
        return self.norm(t)


def patch_vit(monkeypatch):
    # 'vit_*' names get the token stub; anything else gets the pooled CNN stub.
    import timm
    monkeypatch.setattr(timm, 'create_model',
                        lambda name, *a, **k: TinyViT() if name.startswith('vit_') else TinyEncoder())


def test_spread_groups_cover_every_slice():
    assert spread_groups(6) == [[0, 2, 4], [1, 3, 5]]
    assert spread_groups(3) == [[0, 1, 2]]
    assert sorted(i for g in spread_groups(12) for i in g) == list(range(12))
    assert [len(p) for p in slice_thirds(6)] == [2, 2, 2]


def test_slot_query_model(monkeypatch):
    patch_vit(monkeypatch)
    torch.manual_seed(0)
    model = SlotQueryNet('vit_tiny', 2, 3, pretrained=False, img=16, unfreeze_last=1).eval()
    trainable = {n for n, p in model.enc.named_parameters() if p.requires_grad}
    assert trainable == {'blocks.2.weight', 'blocks.2.bias', 'norm.weight', 'norm.bias'}
    x = torch.randint(0, 256, (2, 3, 6, 16, 16), dtype=torch.uint8)
    present = torch.tensor([[True, False, True], [False, False, False]])
    with torch.no_grad():
        full, att = model(x, present)
        # Evaluation over six slices is the mean of its two spread triplets.
        parts = [model(x[:, :, g], present)[0] for g in spread_groups(6)]
        torch.testing.assert_close(full, torch.stack(parts).mean(0))
        # An absent slot's pixels cannot change the prediction.
        y = x.clone(); y[0, 1] = 255 - y[0, 1]
        torch.testing.assert_close(model(y, present)[0][0], full[0])
    assert att.shape == (2, 2, 3) and float(att[0, :, 1].max()) == 0.0
    cfg = dict(arch='slot_query', backbone='vit_tiny', img=16, drop=.2, head_hidden=256, unfreeze_last=1)
    rebuilt = build_model(cfg, 2, 3, pretrained=False)
    rebuilt.load_state_dict(model.state_dict(), strict=True)


def test_real_slot_backbones_forward():
    # Random weights; checks the real timm DINOv2/DINOv3 token layouts end to end.
    for name in ['vit_small_patch14_dinov2.lvd142m', 'vit_small_patch16_dinov3.lvd1689m']:
        model = SlotQueryNet(name, 12, 6, pretrained=False, img=64).eval()
        x = torch.randint(0, 256, (1, 6, 6, 64, 64), dtype=torch.uint8)
        with torch.inference_mode():
            out, _ = model(x, torch.tensor([[True, True, False, True, False, False]]))
        assert out.shape == (1, 12) and torch.isfinite(out).all()


def test_three_families_training_and_mixed_submission(monkeypatch, tmp_path):
    import gc, random, time, glob, traceback
    from torch.utils.data import Dataset, DataLoader
    from tqdm.auto import tqdm
    patch_vit(monkeypatch)
    n = 16; labels = ['a', 'b']; cache_dir = tmp_path / 'cache'; cache_dir.mkdir()
    rng = np.random.default_rng(7)
    np.save(cache_dir / 'cache.npy', rng.integers(0, 256, (n, 6, 24, 16, 16), dtype=np.uint8))
    present = {f'present_{i}': [True] * n if i < 3 else [i % 2 == 0] * n for i in range(6)}
    df = pd.DataFrame(dict(StudyInstanceUID=[f'id{i}' for i in range(n)], source=['pseudo'] * 12 + ['gold'] * 4,
                           row=range(n), cache_id=0, a=[.2, .9] * 8, b=[1, 0, 0, 1] * 4,
                           w_a=[1.] * 12 + [3.] * 4, w_b=[1.] * 12 + [3.] * 4,
                           report_group=[f'g{i // 2}' for i in range(n)], **present))
    meta = pp.preprocessing_meta('knee-2c-v3')
    common = dict(os=os, gc=gc, copy=copy, math=__import__('math'), time=time, json=json, random=random, np=np, pd=pd,
                  torch=torch, nn=nn, F=torch.nn.functional, Dataset=Dataset, DataLoader=DataLoader, tqdm=tqdm,
                  df=df, CACHE_DIRS=[str(cache_dir)], CACHE_META=meta, PLANES=meta['slots'], LABEL_COLS=labels,
                  T_TRAIN=10, NUM_WORKERS=0, DEV='cpu', N_FOLDS=2, SEED=42, GROUP_COL='report_group', N_SLICES=24,
                  NORMALIZATION='pretrained', EPOCHS=1, FOLDS_TO_RUN=[0, 1], RESUME=False, GOLD_WEIGHT=3.,
                  TRAIN_WINDOWS=6, assign_folds=assign_folds, macro_auc=macro_auc, rank_blend=rank_blend,
                  build_model=build_model, slice_thirds=slice_thirds, window_centres=window_centres,
                  GRAD_CHECKPOINT=False, SLOT_DROPOUT=0.0, TEACHER_SUMMARY=None, LABEL_ROOT=str(tmp_path / 'weights'), RAD_VERIFY=False,
                  write_atomic=write_atomic)
    from torchvision.models import resnet50
    (tmp_path / 'weights').mkdir()
    backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    torch.save({f'backbone.{k}': v for k, v in backbone.state_dict().items()}, tmp_path / 'weights' / 'ResNet50.pt')
    families = {
        'reader25d': dict(BACKBONE='cnn_tiny', IMG=16, BATCH=4, ACCUM_STEPS=2, LR=5e-4, ENCODER_LR_SCALE=.2, WD=.02,
                          PCT_START=.1, DROPOUT=.2, UNFREEZE_LAST=None, HEAD_HIDDEN=8, EMA_DECAY=.9, CLIP=3.),
        'slot_query': dict(BACKBONE='vit_tiny', IMG=16, BATCH=4, ACCUM_STEPS=1, LR=1e-3, ENCODER_LR_SCALE=8e-3, WD=.02,
                           PCT_START=.15, DROPOUT=.2, UNFREEZE_LAST=1, HEAD_HIDDEN=8, EMA_DECAY=0., CLIP=5.),
        'kneenet': dict(BACKBONE='cnn_tiny', IMG=16, BATCH=2, ACCUM_STEPS=1, LR=3e-4, ENCODER_LR_SCALE=.3, WD=.01,
                        PCT_START=.1, DROPOUT=.4, UNFREEZE_LAST=None, HEAD_HIDDEN=None, EMA_DECAY=0., CLIP=5.),
        'radhead': dict(BACKBONE='radimagenet_resnet50', IMG=32, BATCH=8, ACCUM_STEPS=1, LR=2e-4, ENCODER_LR_SCALE=0., WD=3e-3,
                        PCT_START=0., DROPOUT=.15, UNFREEZE_LAST=None, HEAD_HIDDEN=16, EMA_DECAY=0., CLIP=1.)}
    outs = {}
    for arch, extra in families.items():
        out = tmp_path / arch; out.mkdir(); outs[arch] = out
        namespace = dict(common, ARCH=arch, PRESET=f'p_{arch}', OUT=str(out), **extra)
        exec(cell('training.ipynb', 9), namespace)
        exec(cell('training.ipynb', 13), namespace)
        if arch == 'radhead':
            exec((ROOT / 'rad_runtime.py').read_text(), namespace)
            assert len(pd.read_csv(out / 'oof_predictions.csv')) == 12
            saved = torch.load(out / 'fold_0.pt', weights_only=False)
            assert saved['cfg']['arch'] == 'radhead' and any(k.startswith('backbone.') for k in saved['model'])
            continue
        x, p, y, w = namespace['KneeDS'](df, np.arange(n), True)[13]
        expect = {'reader25d': (6, 24, 16, 16), 'slot_query': (6, 3, 16, 16), 'kneenet': (6, 10, 16, 16)}[arch]
        assert tuple(x.shape) == expect and float(w[0]) == 3.
        if arch == 'slot_query': assert int(x[3].sum()) == 0      # absent slot stays blank
        exec((ROOT / 'train_runtime.py').read_text(), namespace)
        assert len(pd.read_csv(out / 'oof_predictions.csv')) == 12
        folds = pd.read_csv(out / 'folds.csv')
        assert (folds.loc[folds.source.eq('gold'), 'fold'] == -1).all()
        assert folds.loc[folds.source.eq('pseudo')].assign(g=df.report_group[:12]).groupby('g').fold.nunique().max() == 1
        saved = torch.load(out / 'last_fold_0.pt', weights_only=False)
        assert saved['cfg']['arch'] == arch and saved['cfg']['gold_weight'] == 3.
        assert (saved['ema'] is not None) == (arch == 'reader25d')
    # One run decodes once and rank-blends all three families: they share preprocessing.
    comp = tmp_path / 'competition'; comp.mkdir()
    sample = pd.DataFrame(dict(StudyInstanceUID=['s1', 's0'], a=[0] * 2, b=[0] * 2))
    sample.to_csv(comp / 'sample_submission.csv', index=False)
    manifest = pd.DataFrame(dict(StudyInstanceUID=['s0', 's1'], study_side=['L', 'R'],
                                 **{f'dir_{i}': ['d', 'd' if i < 2 else ''] for i in range(6)}))
    seen = []
    def fake_series(directory, plane, side, **kw):
        seen.append(kw)
        return rng.integers(0, 256, (24, 16, 16), dtype=np.uint8)
    paths = [str(outs[a] / f'fold_{f}.pt') for a in families for f in range(2)]
    cfg = torch.load(paths[0], weights_only=False)['cfg']
    namespace = dict(common, glob=glob, traceback=traceback, manifest=manifest, MODERN=True,
                     SLOTS=pp.RECIPES['knee-2c-v3']['slots'], SERIES_KW=pp.series_kwargs(pp.recipe('knee-2c-v3')),
                     RECIPE=pp.recipe('knee-2c-v3'), slot_kwargs=pp.slot_kwargs,
                     row_anchor=pp.row_anchor, cfg=cfg, CKPT_PATHS=paths,
                     COMP_DIR=str(comp), OUT_CSV=str(tmp_path / 'submission.csv'), MAX_HOURS=1, BATCH=2,
                     IMG=16, load_series=fake_series)
    exec((ROOT / 'inference_runtime.py').read_text(), namespace)
    diagnostics = json.loads((tmp_path / 'inference_diagnostics.json').read_text())
    assert diagnostics['ensemble_members'] == 8
    assert {m['member'].split(':')[0] for m in diagnostics['models']} == set(families)
    assert diagnostics['families'] == {f'p_{a}': 2 for a in families}
    assert seen and all(k['n_slices'] == 24 and k['windows'] == 8 and k['img'] == 240 for k in seen)
    submitted = pd.read_csv(tmp_path / 'submission.csv')
    assert submitted.StudyInstanceUID.tolist() == ['s1', 's0'] and submitted[labels].notna().all().all()
    # Explicit family weights must name every attached family, and are recorded.
    weights = {f'p_{a}': w for a, w in zip(families, (.7, .1, .2, .5))}
    missing = dict(list(weights.items())[:-1])
    with pytest.raises(AssertionError, match='FAMILY_WEIGHTS has no weight'):
        exec((ROOT / 'inference_runtime.py').read_text(), dict(namespace, FAMILY_WEIGHTS=missing))
    exec((ROOT / 'inference_runtime.py').read_text(), dict(namespace, FAMILY_WEIGHTS=weights))
    diagnostics = json.loads((tmp_path / 'inference_diagnostics.json').read_text())
    assert diagnostics['family_weights'] == {f: weights[f] for f in diagnostics['families']}
    # A zero-weight family is skipped before its model is built; the rest still blend.
    first = next(iter(families))
    zeroed = dict(weights, **{f'p_{first}': 0.0})
    exec((ROOT / 'inference_runtime.py').read_text(), dict(namespace, FAMILY_WEIGHTS=zeroed))
    diagnostics = json.loads((tmp_path / 'inference_diagnostics.json').read_text())
    assert f'p_{first}' not in diagnostics['families'] and len(diagnostics['families']) == len(families) - 1
    assert sum(m.get('skipped') == 'weight 0' for m in diagnostics['models']) == 2
    # Transductive self-training: reader members fine-tune on the test volumes, then predict again.
    exec((ROOT / 'inference_runtime.py').read_text(), dict(namespace, FAMILY_WEIGHTS=weights, TTST_EPOCHS=1, TTST_BATCH=2))
    diagnostics = json.loads((tmp_path / 'inference_diagnostics.json').read_text())
    readers = [m for m in diagnostics['ttst']]
    assert readers and all(m['finished'] and m['steps'] == 1 for m in readers)
    assert all('reader' in m['member'] or 'p_reader25d' in m['member'] for m in readers)
    submitted = pd.read_csv(tmp_path / 'submission.csv')
    assert submitted[labels].notna().all().all()


def test_reader_model(monkeypatch):
    patch_vit(monkeypatch)
    torch.manual_seed(0)
    model = Reader25D('cnn_tiny', 2, 3, pretrained=False, img=16, dim=8, train_windows=2).eval()
    x = torch.randint(0, 256, (2, 3, 12, 16, 16), dtype=torch.uint8)
    present = torch.tensor([[True, False, True], [False, False, False]])
    with torch.no_grad():
        full, att = model(x, present)
        y = x.clone(); y[0, 1] = 255 - y[0, 1]
        torch.testing.assert_close(model(y, present)[0][0], full[0])   # absent slot ignored
    assert att.shape == (2, 12, 2) and float(att[0, 4:8].max()) == 0.0
    model.train()
    out, att = model(x, present)
    assert int((att[0].sum(-1) > 0).sum()) == 2 * 2                  # 2 windows x 2 present slots
    out.sum().backward()


def test_label_builder(tmp_path):
    labels = ['ACL', 'MCL']
    train = pd.DataFrame(dict(StudyInstanceUID=['s0', 's1', 's2', 'g0'], Report=['Same text.', 'same  TEXT', 'other', 'x'],
                              ACL=[np.nan, np.nan, np.nan, 1.], MCL=[np.nan, np.nan, np.nan, 0.]))
    root = tmp_path / 'input'
    for d in ('pk', 'qw', 'own', 'labels01/labels'): (root / d).mkdir(parents=True)
    (root / 'comp' / 'train_series' / 'x').mkdir(parents=True)
    pd.DataFrame(dict(StudyInstanceUID=['s0', 's1', 's2', 'g0'], ACL=[1., 0., .5, 0.], MCL=[0., 0., 1., 1.])).to_csv(root / 'pk' / 'report_labels_v2.csv', index=False)
    pd.DataFrame({'StudyInstanceUID': ['s0', 's2'], 'ACL__label': [0., 1.5], 'MCL__label': [0., 1.]}).to_csv(root / 'qw' / 'qwen_knee_weak_labels.csv', index=False)
    pd.DataFrame(dict(StudyInstanceUID=['s1'], ACL=[1.], MCL=[1.])).to_csv(root / 'own' / 'extra_labels_sol_v2.csv', index=False)
    # 01_labels output: the Flash member file is a reader; consensus and other members are not.
    flash = [dict(StudyInstanceUID='s1', acl='possible', mcl='unsure'), dict(StudyInstanceUID='s2', acl='probable', mcl='absent')]
    (root / 'labels01/labels/pseudo_google_gemini_3_1_flash_lite_preview.jsonl').write_text('\n'.join(json.dumps(r) for r in flash))
    (root / 'labels01/labels/pseudo_anthropic_claude_sonnet_5_default.jsonl').write_text('{}')
    (root / 'labels01/labels/validate_google_gemini_3_1_flash_lite_preview.jsonl').write_text('{}')
    pd.DataFrame(dict(StudyInstanceUID=['s1'], ACL=[1.], MCL=[1.], source=['pseudo'])).to_csv(root / 'labels01' / 'train_targets.csv', index=False)
    tables = lb.find_label_tables(str(root))
    assert set(tables) == {'report_labels_v2.csv', 'qwen_knee_weak_labels.csv', 'extra_labels_sol_v2.csv',
                           'pseudo_google_gemini_3_1_flash_lite_preview.jsonl'}
    frame, report = lb.build_targets(train, labels, tables, {'extra_labels_sol_v2.csv': 3.}, gold_weight=3.)
    t = frame.set_index('StudyInstanceUID')
    assert t.loc['s0', 'ACL'] == pytest.approx(.5)                       # mean of 1 and 0
    assert t.loc['s1', 'ACL'] == pytest.approx((0 + 3 * 1 + .35) / 5)     # Sol weighted 3x, Flash 'possible' = .35
    assert t.loc['s1', 'MCL'] == pytest.approx(3 / 4)                     # Flash 'unsure' abstains
    assert t.loc['s2', 'ACL'] == pytest.approx((.5 + .75) / 2)            # qwen 1.5 ignored; Flash 'probable'
    assert t.loc['s2', 'MCL'] == pytest.approx(2 / 3)
    assert t.loc['g0', 'ACL'] == 1. and t.loc['g0', 'w_ACL'] == 3. and t.loc['g0', 'source'] == 'gold'
    assert t.loc['s0', 'report_group'] == t.loc['s1', 'report_group'] != t.loc['s2', 'report_group']
    assert 'consensus' in report
    shard = tmp_path / 'broken.jsonl'
    shard.write_text(json.dumps(flash[0]) + '\\n' + json.dumps(flash[1]))   # literal backslash-n separator
    assert len(list(lb.iter_jsonl(shard))) == 2

def test_family_blend_is_per_family(monkeypatch, tmp_path):
    # Three identical members of one family must not outvote one member of another.
    a = np.array([[.1], [.2], [.9]]); b = np.array([[.9], [.8], [.1]])
    ns = dict(np=np, pd=pd, rank_blend=rank_blend, all_predictions=[a, a, a, b],
              member_families=['f1', 'f1', 'f1', 'f2'], manifest=pd.DataFrame(index=range(3)), LABEL_COLS=['x'])
    src = (ROOT / 'inference_runtime.py').read_text()
    block = src[src.index('family_weights = '):src.index('fill = np.median(preds, 0)')]
    exec(block, ns)
    np.testing.assert_allclose(ns['preds'][:, 0], [2/3, 2/3, 2/3])   # member-weighted would be [.5, .67, .83]


def test_run_folder_mirrors_launchers():
    run = ROOT / 'KAGGLE_RUN'
    copies = sorted(run.rglob('*.ipynb'))
    launchers = sorted((ROOT / 'RUN_ON_KAGGLE').glob('*.ipynb'))
    assert sorted(p.name for p in copies) == sorted(p.name for p in launchers)   # each exactly once
    for copy in copies:
        assert copy.read_bytes() == (ROOT / 'RUN_ON_KAGGLE' / copy.name).read_bytes()
    uploads = sorted(p.name for p in (run / 'step1_upload_dataset' / 'reader_labels').glob('*.csv'))
    assert uploads == ['extra_labels_sol_v2.csv', 'extra_labels_sonnet_legacy.csv']   # no Gemini 3.1 Pro
    assert (run / 'README.md').is_file()


SITECUSTOMIZE = """
import timm, torch, torch.nn as nn
class _CNN(nn.Module):
    num_features = 4
    pretrained_cfg = {'input_size': (3, 32, 32), 'mean': (.485, .456, .406), 'std': (.229, .224, .225)}
    def __init__(self): super().__init__(); self.conv = nn.Conv2d(3, 4, 1)
    def forward(self, x): return self.conv(x).mean((-2, -1))
class _ViT(nn.Module):
    num_features, num_prefix_tokens = 4, 1
    pretrained_cfg = _CNN.pretrained_cfg
    def __init__(self):
        super().__init__(); self.patch = nn.Conv2d(3, 4, 8, 8); self.cls = nn.Parameter(torch.zeros(1, 1, 4))
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)]); self.norm = nn.LayerNorm(4)
    def forward_features(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        for b in self.blocks: t = t + b(t)
        return self.norm(t)
timm.create_model = lambda name, *a, **k: _ViT() if name.startswith('vit_') else _CNN()
"""


def fake_kaggle_inputs(root, n=40, seed=3, version='knee-2c-v3'):
    labels = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
              "Effusion","Synovitis","Baker's","Contusion","Fracture"]
    rng = np.random.default_rng(seed); ids = [f'1.2.{i}' for i in range(n)]
    comp = root / 'competition'; comp.mkdir(parents=True)
    train = pd.DataFrame({'StudyInstanceUID': ids, 'Report': [f'r{i % 30}' for i in range(n)]})
    for c in labels: train[c] = np.where(np.arange(n) >= n - 6, rng.integers(0, 2, n), np.nan)
    train.to_csv(comp / 'train.csv', index=False)
    meta = dict(pp.preprocessing_meta(version), img=16)
    n_slots = len(meta['slots'])
    for shard in range(2):
        d = root / f'shard{shard}'; d.mkdir(); rows = list(range(shard, n, 2))
        json.dump(dict(meta, shard=shard, n_shards=2), open(d / 'cache_meta.json', 'w'))
        np.save(d / 'cache.npy', rng.integers(0, 256, (len(rows), n_slots, 24, 16, 16), dtype=np.uint8))
        idx = pd.DataFrame({'StudyInstanceUID': [ids[r] for r in rows], 'study_side': 'L', 'side_src': 'tag',
                            'shard': shard, 'row': range(len(rows))})
        for i in range(n_slots): idx[f'present_{i}'] = i < 4 or i >= 6
        idx['n_present'] = 4
        idx.to_csv(d / 'cache_index.csv', index=False)
    for name, suffix in [('report_labels_v2.csv', ''), ('llm_labels_v4_blend.csv', ''),
                         ('labels_llm_gpt56sol.csv', ''), ('qwen_knee_weak_labels.csv', '__label'),
                         ('extra_labels_sol_v2.csv', ''), ('extra_labels_sonnet_legacy.csv', '')]:
        d = root / name.split('.')[0]; d.mkdir()
        pd.DataFrame({'StudyInstanceUID': ids, **{c + suffix: rng.choice([0., 1., .5], n) for c in labels}}).to_csv(d / name, index=False)
    flash = root / 'labels01' / 'labels'; flash.mkdir(parents=True)
    (flash / 'pseudo_google_gemini_3_1_flash_lite_preview.jsonl').write_text(
        '\n'.join(json.dumps({'StudyInstanceUID': u, 'acl': 'definite', 'mcl': 'absent'}) for u in ids[:30]))
    return comp


def run_session(notebook, tmp_path, monkeypatch, plan=None, extra=None, version='knee-2c-v3'):
    root = tmp_path / 'input'
    comp = fake_kaggle_inputs(root, version=version)
    if extra is not None:
        extra(root)
    site = tmp_path / 'site'; site.mkdir(); (site / 'sitecustomize.py').write_text(SITECUSTOMIZE)
    for key, value in dict(KNEE_INPUT_ROOT=str(root), KNEE_COMP_DIR=str(comp), KNEE_ALLOW_CPU='1',
                           PYTHONPATH=str(site)).items():
        monkeypatch.setenv(key, value)
    cells = [''.join(c['source']) for c in json.loads((ROOT / 'RUN_ON_KAGGLE' / notebook).read_text())['cells']
             if c['cell_type'] == 'code']
    ns = dict(WORK=str(tmp_path / 'working'), POLL_SECONDS=1, REQUIRE_GPUS=False, WORKERS_PER_JOB=0)
    for code in cells[1:]:                     # cells[0] installs packages on Kaggle
        exec(code, ns)
        if plan is not None and 'GPU_JOBS = ' in code: ns['GPU_JOBS'] = plan
    return tmp_path / 'working', ns


def test_two_gpu_session_trains_both_folds(tmp_path, monkeypatch):
    work, ns = run_session('04_session_A_reader_f0_f1.ipynb', tmp_path, monkeypatch)
    out = work / 'reader_convnext'
    assert (out / 'fold_0.pt').is_file() and (out / 'fold_1.pt').is_file()
    results = json.loads((work / 'session_results.json').read_text())['jobs']
    assert {v['returncode'] for v in results.values()} == {0} and len(results) == 2
    log = (work / 'logs' / 'gpu0_reader_convnext_f0.log').read_text()
    assert 'pseudo_google_gemini_3_1_flash_lite_preview.jsonl' in log and "'epoch'" in log
    cfg = torch.load(out / 'fold_1.pt', weights_only=False)['cfg']
    assert cfg['preset'] == 'reader_convnext' and pd.read_csv(out / 'folds.csv').fold.max() == 4


def test_two_gpu_session_reports_failed_job(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match='failed'):
        run_session('04_session_A_reader_f0_f1.ipynb', tmp_path, monkeypatch,
                    plan={0: [('no_such_preset', [0])], 1: [('reader_convnext', [1])]})


def test_input_check_runs_job_data_setup(tmp_path, monkeypatch, capsys):
    # The CPU check uses the jobs' own setup, so a pass means a session would find its inputs.
    root = tmp_path / 'input'
    comp = fake_kaggle_inputs(root)
    for key, value in dict(KNEE_INPUT_ROOT=str(root), KNEE_COMP_DIR=str(comp), KNEE_OUT=str(tmp_path / 'out')).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('KNEE_ALLOW_CPU', raising=False)
    cells = [''.join(c['source']) for c in json.loads((ROOT / 'RUN_ON_KAGGLE' / '04_input_check.ipynb').read_text())['cells']
             if c['cell_type'] == 'code']
    ns = {}
    for code in cells[1:]:                     # cells[0] installs packages on Kaggle
        exec(code, ns)
    out = capsys.readouterr().out
    assert 'pseudo_google_gemini_3_1_flash_lite_preview.jsonl' in out and 'MB/s' in out
    assert 'not attached (needed by session C only)' in out and out.rstrip().endswith('INPUT CHECK PASSED: caches, label readers and gold rows are as training expects.')


def test_input_check_reads_cache_pieces(tmp_path, monkeypatch, capsys):
    # Kaggle attaches ~1 GB pieces, each a complete shard; training must see every study once.
    root = tmp_path / 'input'
    comp = fake_kaggle_inputs(root)
    for shard in range(2):
        source = root / f'shard{shard}'
        cache, index = np.load(source / 'cache.npy'), pd.read_csv(source / 'cache_index.csv')
        meta = json.load(open(source / 'cache_meta.json'))
        bounds = np.linspace(0, len(index), 4).astype(int)
        for j in range(3):
            piece = root / f'pieces{shard}' / f'cache_piece_{shard * 3 + j:02d}'; piece.mkdir(parents=True)
            np.save(piece / 'cache.npy', cache[bounds[j]:bounds[j + 1]])
            index.iloc[bounds[j]:bounds[j + 1]].assign(shard=shard * 3 + j, row=lambda d: np.arange(len(d))).to_csv(
                piece / 'cache_index.csv', index=False)
            json.dump(dict(meta, shard=shard * 3 + j, n_shards=6), open(piece / 'cache_meta.json', 'w'))
        shutil.rmtree(source)
    for key, value in dict(KNEE_INPUT_ROOT=str(root), KNEE_COMP_DIR=str(comp), KNEE_OUT=str(tmp_path / 'out')).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('KNEE_ALLOW_CPU', raising=False)
    cells = [''.join(c['source']) for c in json.loads((ROOT / 'RUN_ON_KAGGLE' / '04_input_check.ipynb').read_text())['cells']
             if c['cell_type'] == 'code']
    ns = {}
    for code in cells[1:]:
        exec(code, ns)
    assert len(ns['CACHE_DIRS']) == 6 and len(ns['df']) == 40 and ns['df']['StudyInstanceUID'].is_unique
    assert 'INPUT CHECK PASSED' in capsys.readouterr().out


def test_cache_writes_mountable_pieces(monkeypatch, tmp_path):
    # PIECES > 1: each piece is a complete shard (pixels, index, meta) with global shard ids.
    monkeypatch.setattr(pp, 'load_series', lambda d, *a, **k: np.full((2, 8, 8), ord(d[-1]), np.uint8))
    import multiprocessing as mp
    monkeypatch.setattr(mp, 'get_all_start_methods', lambda: [])
    meta = dict(pp.preprocessing_meta('knee-2c-v3'), img=8, n_slices=2)
    manifest = pd.DataFrame([dict(StudyInstanceUID=f's{i}', study_side='L', side_src='tag', dir_0=f'd{i}', dir_1='',
                                  dir_2='', dir_3='', dir_4='', dir_5='') for i in range(7)])
    from tqdm.auto import tqdm
    namespace = dict(vars(pp), manifest=manifest, WORK=str(tmp_path), CACHE=str(tmp_path / 'cache.npy'),
                     INDEX=str(tmp_path / 'cache_index.csv'), METAJ=str(tmp_path / 'cache_meta.json'), IMG=8,
                     N_SLICES=2, tqdm=tqdm, preprocessing_meta=lambda: meta, N_SHARDS=2, SHARD=1, PIECES=3,
                     SLOTS=pp.RECIPES['knee-2c-v3']['slots'], recipe=lambda: pp.recipe('knee-2c-v3'))
    code = (ROOT / 'cache_runtime.py').read_text()
    exec(code, namespace)
    namespace['manifest'] = manifest
    exec(code, namespace)                                   # resume: nothing left to do, same pieces
    pieces = sorted(tmp_path.glob('cache_piece_*'))
    assert [p.name for p in pieces] == ['cache_piece_03', 'cache_piece_04', 'cache_piece_05']
    assert not (tmp_path / 'cache.npy').exists()
    ids = []
    for p in pieces:
        m, idx, arr = json.load(open(p / 'cache_meta.json')), pd.read_csv(p / 'cache_index.csv'), np.load(p / 'cache.npy')
        assert m['n_shards'] == 6 and (idx['shard'] == m['shard']).all() and idx['row'].tolist() == list(range(len(idx)))
        for r, sid in zip(idx['row'], idx['StudyInstanceUID']):
            assert (arr[r, 0] == ord(sid[-1])).all() and (arr[r, 1:] == 0).all()
        ids += idx['StudyInstanceUID'].tolist()
    assert ids == ['s1', 's3', 's5']                        # shard 1 of 2 holds every second study, in order


def test_session_trains_extra_seed_of_a_family(tmp_path, monkeypatch):
    # A second seed of the same family-fold trains in its own folder and records its seed.
    work, ns = run_session('04_session_A_reader_f0_f1.ipynb', tmp_path, monkeypatch,
                           plan={0: [('reader_convnext', [0])], 1: [('reader_convnext', [0], 7)]})
    base, extra = (torch.load(work / d / 'fold_0.pt', weights_only=False)['cfg'] for d in ('reader_convnext', 'reader_convnext_s7'))
    assert base['seed'] == 42 and extra['seed'] == 7 and base['preset'] == extra['preset'] == 'reader_convnext'
    assert set(json.loads((work / 'session_results.json').read_text())['jobs']) == {
        'gpu0_reader_convnext_f0', 'gpu1_reader_convnext_s7_f0'}
    sub = ''.join(json.loads((ROOT / 'RUN_ON_KAGGLE' / '05_submit.ipynb').read_text())['cells'][1]['source'])
    assert "other_ck.get('fold', path), other.get('seed'))" in sub        # seeds may share a family-fold


def test_raptor_blend_cell(tmp_path):
    # The final cell rank-blends the public Raptor member with our ensemble, and falls back
    # to our ensemble alone if the Raptor output is missing or does not match.
    nb = json.loads((ROOT / 'RUN_ON_KAGGLE' / '05b_submit_raptor_blend_w30.ipynb').read_text())
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert 'RAPTOR_SOURCE = ' in code[0] and 'OWN_WEIGHT = 0.3' in code[0]
    assert 'own_submission.csv' in code[1] and 'RAPTOR_HOURS' in code[1]
    blend = code[-1]
    ids = ['s1', 's2', 's3', 's4']
    own = pd.DataFrame({'StudyInstanceUID': ids, 'ACL': [.1, .2, .3, .4], 'MCL': [.4, .3, .2, .1]})
    rap = pd.DataFrame({'StudyInstanceUID': ids[::-1], 'ACL': [.9, .8, .7, .6], 'MCL': [.1, .2, .3, .4]})
    own.to_csv(tmp_path / 'own.csv', index=False); rap.to_csv(tmp_path / 'rap.csv', index=False)
    blend = blend.replace("'/kaggle/working/submission.csv'", repr(str(tmp_path / 'submission.csv')))
    ns = dict(os=os, OUT_CSV=str(tmp_path / 'own.csv'), RAPTOR_OUT=str(tmp_path / 'rap.csv'), OWN_WEIGHT=.3)
    exec(blend, ns)
    out = pd.read_csv(tmp_path / 'submission.csv')
    assert out.StudyInstanceUID.tolist() == ids and 'Raptor x 0.70' in ns['used']
    # rap reversed row order: s4=.9 ... s1=.6 for ACL, so Raptor ACL ranks agree with ours
    assert out.ACL.is_monotonic_increasing
    for bad in ('missing', 'rows'):
        if bad == 'missing': os.remove(tmp_path / 'rap.csv')
        else: rap.iloc[:3].to_csv(tmp_path / 'rap.csv', index=False)
        exec(blend, ns)
        assert ns['used'] == 'our ensemble alone'
        assert pd.read_csv(tmp_path / 'submission.csv')[['ACL', 'MCL']].equals(own[['ACL', 'MCL']])


def test_oof_blend_check_scores_families_without_gold(tmp_path):
    import subprocess, sys
    labels = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
              "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
    rng = np.random.default_rng(0)
    n = 200
    ids = [f'1.2.{i}' for i in range(n)]
    y = rng.random((n, 12))
    targets = pd.DataFrame(y, columns=labels)
    targets.insert(0, 'StudyInstanceUID', ids)
    targets['source'] = np.where(np.arange(n) < 10, 'gold', 'pseudo')
    fold = np.where(targets['source'].eq('gold'), -1, np.arange(n) % 5)

    def write(session, family, folds, noise):
        out = tmp_path / session / family
        out.mkdir(parents=True)
        targets.to_csv(out / 'targets_used.csv', index=False)
        for f in folds:
            rows = np.flatnonzero(fold == f)
            pred = pd.DataFrame(y[rows] + rng.normal(0, noise, (len(rows), 12)), columns=labels)
            pred.insert(0, 'StudyInstanceUID', [ids[i] for i in rows])
            pred['fold'] = f
            pred.to_csv(out / f'oof_fold_{f}.csv', index=False)
    write('a', 'reader_convnext', [0, 1, 2], .2)
    write('b', 'reader_convnext', [3, 4], .2)
    write('b', 'slot_dinov2', range(5), 1.0)
    run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'oof_blend_check.py'), str(tmp_path)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    assert "Complete families: ['reader_convnext', 'slot_dinov2']" in run.stdout
    top = run.stdout.split('Top family weights')[1].splitlines()[2].split()
    assert float(top[0]) > float(top[1])       # the cleaner family earns the larger weight
    assert 'OOF BLEND CHECK DONE' in run.stdout
    # A study that is out-of-fold twice (the same fold attached twice) stops the check.
    write('c', 'radimagenet', [0], .5)
    write('d', 'radimagenet', [0], .5)
    run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'oof_blend_check.py'), str(tmp_path)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode != 0 and 'OOF in two attached folds' in run.stderr


def test_caformer_reader_preset_builds_and_trains_a_step():
    # The diversity reader shares Reader25D; only the timm backbone and the micro-batch change.
    import re
    from model_core import Reader25D
    text = (ROOT / 'train_config.py').read_text()
    block = text[text.index("'reader_caformer'"):text.index("'reader_coatnet'")]
    assert "caformer_s18.sail_in22k_ft_in1k" in block and 'batch=2, accum=4' in block
    model = Reader25D('caformer_s18', 12, 6, pretrained=False, img=64, dim=32, train_windows=2).train()
    x = torch.randint(0, 255, (1, 6, 12, 64, 64), dtype=torch.uint8)
    present = torch.tensor([[True, True, False, True, False, False]])
    logits, attention = model(x, present)
    assert logits.shape == (1, 12) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(p.grad is not None for p in model.enc.parameters() if p.requires_grad)



LABELS12 = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
            "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def test_teacher_targets_average_our_readers_out_of_fold(tmp_path):
    import subprocess, sys
    rng = np.random.default_rng(1)
    n = 60
    ids = [f'1.3.{i}' for i in range(n)]
    targets = pd.DataFrame(rng.random((n, 12)), columns=LABELS12)
    targets.insert(0, 'StudyInstanceUID', ids)
    targets['source'] = np.where(np.arange(n) < 4, 'gold', 'pseudo')
    pseudo_rows = np.arange(4, n)
    preds = {}
    def write(session, folder, folds, split):
        out = tmp_path / 'in' / session / folder
        out.mkdir(parents=True)
        targets.to_csv(out / 'targets_used.csv', index=False)
        for f in folds:
            rows = pseudo_rows[split[pseudo_rows] == f]
            p = rng.random((len(rows), 12))
            preds.setdefault(folder, {}).update({ids[r]: p[k] for k, r in enumerate(rows)})
            frame = pd.DataFrame(p, columns=LABELS12)
            frame.insert(0, 'StudyInstanceUID', [ids[r] for r in rows]); frame['fold'] = f
            frame.to_csv(out / f'oof_fold_{f}.csv', index=False)
    write('a', 'reader_convnext', range(5), np.arange(n) % 5)
    write('d', 'reader_convnext_s7', [0, 1], (np.arange(n) // 2) % 5)
    write('c', 'slot_dinov2', range(5), np.arange(n) % 5)          # not a teacher family
    run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'teacher_targets.py'), str(tmp_path / 'in')],
                         capture_output=True, text=True, env=dict(os.environ, TEACHER_OUT=str(tmp_path / 'out')))
    assert run.returncode == 0, run.stderr
    teacher = pd.read_csv(tmp_path / 'out' / 'teacher_targets.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert not set(ids[:4]) & set(teacher.index)                    # gold never gets a teacher value
    assert len(teacher) == n - 4
    for sid in teacher.index:
        members = [preds[m][sid] for m in ('reader_convnext', 'reader_convnext_s7') if sid in preds[m]]
        np.testing.assert_allclose(teacher.loc[sid, LABELS12].to_numpy(float), np.mean(members, axis=0), rtol=1e-6)
        assert teacher.loc[sid, 'n_members'] == len(members)
    meta = json.loads((tmp_path / 'out' / 'teacher_meta.json').read_text())
    assert set(meta['members']) == {'reader_convnext', 'reader_convnext_s7'}


def test_apply_teacher_mixes_pseudo_rows_only(tmp_path):
    from label_builder import apply_teacher, find_teacher
    cols = ['a', 'b']
    frame = pd.DataFrame({'StudyInstanceUID': ['g', 'p1', 'p2', 'p3'], 'a': [1., .2, np.nan, .8], 'b': [0., .9, .5, np.nan],
                          'w_a': [3., 1., 0., 1.], 'w_b': [3., 1., 1., 0.], 'source': ['gold', 'pseudo', 'pseudo', 'pseudo']})
    teacher = pd.DataFrame({'StudyInstanceUID': ['g', 'p1', 'p2'], 'a': [.0, .6, .3], 'b': [.9, .1, np.nan]})
    (tmp_path / 'sub').mkdir()
    teacher.to_csv(tmp_path / 'sub' / 'teacher_targets.csv', index=False)
    path = find_teacher(str(tmp_path))
    out, summary = apply_teacher(frame, path, cols, .6)
    g, p1, p2, p3 = (out.iloc[i] for i in range(4))
    assert (g.a, g.b, g.w_a, g.w_b) == (1., 0., 3., 3.)                       # gold untouched
    assert np.isclose(p1.a, .4 * .2 + .6 * .6) and np.isclose(p1.b, .4 * .9 + .6 * .1)
    assert np.isclose(p2.a, .3) and np.isclose(p2.w_a, .6)                    # unlabelled cell takes the teacher
    assert np.isclose(p2.b, .5)                                               # no teacher value: report kept
    assert np.isclose(p3.a, .8) and np.isnan(p3.b) and p3.w_b == 0            # study without teacher unchanged
    assert summary['mixed_cells'] == 2 and summary['filled_cells'] == 1 and summary['studies'] == 2
    (tmp_path / 'dup').mkdir(); teacher.to_csv(tmp_path / 'dup' / 'teacher_targets.csv', index=False)
    with pytest.raises(AssertionError, match='one teacher table'):
        find_teacher(str(tmp_path))


def test_teacher_gate_follows_report_agreement(tmp_path):
    from label_builder import apply_teacher
    frame = pd.DataFrame({'StudyInstanceUID': ['p1', 'p2'], 'a': [.01, .5], 'w_a': [1., 1.], 'source': ['pseudo'] * 2})
    pd.DataFrame({'StudyInstanceUID': ['p1', 'p2'], 'a': [.9, .9]}).to_csv(tmp_path / 'teacher_targets.csv', index=False)
    out, summary = apply_teacher(frame, str(tmp_path / 'teacher_targets.csv'), ['a'], (.5, .9))
    share_confident = .5 + .4 * (1 - abs(2 * .01 - 1))          # readers agree: mostly the lower share
    assert np.isclose(out.a[0], (1 - share_confident) * .01 + share_confident * .9)
    assert np.isclose(out.a[1], .1 * .5 + .9 * .9)                # split readers: share 0.9
    assert summary['weight'] == [.5, .9]


def test_reader_slot_dropout_trains_only(monkeypatch):
    from model_core import Reader25D
    patch_vit(monkeypatch)
    torch.manual_seed(0)
    model = Reader25D('cnn_tiny', 2, 4, pretrained=False, img=16, dim=8, train_windows=0, slot_dropout=.9)
    seen = []
    model.enc.register_forward_hook(lambda m, i, o: seen.append(i[0].shape[0]))
    x = torch.randint(0, 255, (3, 4, 6, 16, 16), dtype=torch.uint8)
    present = torch.tensor([[True, True, True, False], [True, False, False, False], [True, True, True, True]])
    model.train()
    for _ in range(5): model(x, present)
    # 8 present series x 2 windows = 16 images; every study keeps at least one series (>= 6 images).
    assert all(6 <= n < 16 for n in seen)
    seen.clear(); model.eval()
    with torch.no_grad(): model(x, present)
    assert seen == [16]                                     # inference encodes every present series


def test_student_builds_teacher_in_session_when_none_attached(tmp_path):
    # The train_data teacher block: with reader OOF outputs attached and no teacher table, the
    # session builds the teacher itself, saves it, and mixes it into the pseudo rows only.
    import label_builder as lb
    src = (ROOT / 'train_data.py').read_text()
    block = src[src.index('TEACHER_SUMMARY = None'):src.index("write_atomic(os.path.join(OUT, 'targets_used.csv')")]
    cols = LABELS12
    ids = [f'9.{i}' for i in range(12)]
    targets = pd.DataFrame(np.full((12, 12), .2), columns=cols)
    targets.insert(0, 'StudyInstanceUID', ids)
    for c in cols: targets[f'w_{c}'] = 1.0
    targets['source'] = ['gold'] * 2 + ['pseudo'] * 10
    out = tmp_path / 'in' / 'sess' / 'reader_convnext'
    out.mkdir(parents=True)
    oof = pd.DataFrame(np.full((10, 12), .8), columns=cols)
    oof.insert(0, 'StudyInstanceUID', ids[2:]); oof['fold'] = np.arange(10) % 5
    oof.to_csv(out / 'oof_fold_0.csv', index=False)
    def write_atomic(path, writer): writer(path)
    ns = dict(vars(lb), np=np, pd=pd, os=os, json=json, targets=targets, LABEL_COLS=cols,
              LABEL_ROOT=str(tmp_path / 'in'), OUT=str(tmp_path / 'out'), TEACHER_WEIGHT=(.5, .9),
              write_atomic=write_atomic)
    exec(block, ns)
    assert (tmp_path / 'out' / 'teacher_targets_used.csv').exists()
    mixed = ns['targets']
    a = .5 + .4 * (1 - abs(2 * .2 - 1))
    assert np.allclose(mixed.loc[2:, cols].to_numpy(float), (1 - a) * .2 + a * .8)
    assert np.allclose(mixed.loc[:1, cols].to_numpy(float), .2)          # gold rows untouched
    assert ns['TEACHER_SUMMARY']['studies'] == 10


def test_kaggle_push_metadata_pins_environment():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    import kaggle_push as kp
    meta = kp.metadata('knee-x', 'n.ipynb', None, True, False, ['knee-session-m-student-f0-f1', 'other/slug'],
                       ['knee-wheelhouse'], True)
    assert meta['id'] == 'shreyanshgoyaliitb/knee-x' and meta['kernel_type'] == 'notebook'
    assert meta['kernel_sources'] == ['shreyanshgoyaliitb/knee-session-m-student-f0-f1', 'other/slug']
    assert meta['dataset_sources'] == ['shreyanshgoyaliitb/knee-wheelhouse']
    assert meta['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['docker_image'].startswith('gcr.io/')
    assert meta['enable_internet'] is False and meta['enable_gpu'] is True
    tpu = kp.metadata('knee-t', 'n.ipynb', None, False, True, [], [], True, tpu=True)
    assert tpu['enable_tpu'] is True and tpu['enable_gpu'] is False and tpu['machine_shape'] == 'TpuV5E8'
    assert 'docker_image' not in tpu          # TPU image ships torch_xla; the pinned image is GPU-only


def test_round2_teacher_includes_students_only_when_asked(tmp_path):
    from label_builder import build_teacher
    ids = [f'5.{i}' for i in range(10)]
    for folder, value in [('reader_convnext', .2), ('reader_student', .8), ('slot_dinov2', .5)]:
        out = tmp_path / folder
        out.mkdir()
        frame = pd.DataFrame(np.full((10, 12), value), columns=LABELS12)
        frame.insert(0, 'StudyInstanceUID', ids); frame['fold'] = np.arange(10) % 5
        frame.to_csv(out / 'oof_fold_0.csv', index=False)
    round1, _ = build_teacher(str(tmp_path), LABELS12)
    assert np.allclose(round1[LABELS12].to_numpy(float), .2)            # students are not round-1 teachers
    round2, meta = build_teacher(str(tmp_path), LABELS12, families={'reader_convnext': 1.0, 'reader_student': 1.0})
    assert np.allclose(round2[LABELS12].to_numpy(float), .5) and set(meta['families']) == {'reader_convnext', 'reader_student'}


def test_compact_teacher_merges_shards_and_requires_its_inputs(tmp_path):
    # reader_student_ct_v5: the compact model's OOF arrives as two shard outputs (disjoint studies, same folder
    # name) and weighs 2.0 against our two reader families; a session without those outputs must stop.
    import label_builder as lb
    tree = ast.parse((ROOT / 'train_config.py').read_text())      # train_config itself needs a cache
    presets = next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'PRESETS')
    call = presets.values[[ast.literal_eval(k) for k in presets.keys].index('reader_student_ct_v5')]
    families = ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'teacher_families'))
    assert families == {'reader_convnext': 1.0, 'reader_compact': 1.0}
    ids = [f'7.{i}' for i in range(10)]
    for folder, value, rows in [('a/reader_convnext', .2, ids), ('b/reader_caformer', .4, ids),
                                ('s0/reader_compact', .9, ids[:5]), ('s1/reader_compact', .9, ids[5:])]:
        out = tmp_path / 'in' / folder
        out.mkdir(parents=True)
        frame = pd.DataFrame(np.full((len(rows), 12), value), columns=LABELS12)
        frame.insert(0, 'StudyInstanceUID', rows); frame['fold'] = np.arange(len(rows)) % 4
        frame.to_csv(out / 'oof_fold_0.csv', index=False)
    teacher, meta = lb.build_teacher(str(tmp_path / 'in'), LABELS12, families=families)
    assert np.allclose(teacher[LABELS12].to_numpy(float), (.2 + .9) / 2)       # reader_caformer is not a teacher family here
    assert meta['members']['reader_compact']['studies'] == 10
    src = (ROOT / 'train_data.py').read_text()
    block = src[src.index('TEACHER_SUMMARY = None'):src.index("write_atomic(os.path.join(OUT, 'targets_used.csv')")]
    targets = pd.DataFrame(np.full((10, 12), .2), columns=LABELS12)
    targets.insert(0, 'StudyInstanceUID', ids)
    for c in LABELS12: targets[f'w_{c}'] = 1.0
    targets['source'] = 'pseudo'
    def write_atomic(path, writer): writer(path)
    ns = dict(vars(lb), np=np, pd=pd, os=os, json=json, targets=targets, LABEL_COLS=LABELS12,
              LABEL_ROOT=str(tmp_path / 'in' / 'a'), OUT=str(tmp_path / 'out'), TEACHER_WEIGHT=(.5, .9),
              TEACHER_FAMILY_WEIGHTS=families, write_atomic=write_atomic)
    with pytest.raises(AssertionError, match='reader_compact'):
        exec(block, ns)


def test_quantile_map_keeps_outside_ranking_on_our_scale(tmp_path):
    import label_builder as lb
    ids = [f'8.{i}' for i in range(5)]
    ours = np.array([.1, .2, .3, .4, .5])
    outside = np.array([.99, .7, .9, .8, .95])          # other scale, other order
    for folder, values in [('reader_convnext', ours), ('reader_compact', outside)]:
        out = tmp_path / folder
        out.mkdir()
        frame = pd.DataFrame(np.repeat(values[:, None], 12, axis=1), columns=LABELS12)
        frame.insert(0, 'StudyInstanceUID', ids); frame['fold'] = 0
        frame.to_csv(out / 'oof_fold_0.csv', index=False)
    teacher, meta = lb.build_teacher(str(tmp_path), LABELS12, families={'reader_convnext': 1.0, 'reader_compact': 1.0},
                                     quantile_map=('reader_compact',))
    mapped = np.quantile(ours, (pd.Series(outside).rank().to_numpy() - .5) / 5)
    assert np.allclose(teacher['ACL'].to_numpy(), (ours + mapped) / 2)
    assert sorted(mapped) == pytest.approx(sorted(np.quantile(ours, (np.arange(5) + .5) / 5)))
    assert meta['quantile_mapped'] == ['reader_compact']


def test_static_reader_matches_dynamic_and_cpu_augmentation(monkeypatch):
    from model_core import Reader25D, augment_series_cpu
    patch_vit(monkeypatch)
    torch.manual_seed(0)
    model = Reader25D('cnn_tiny', 3, 4, pretrained=False, img=16, dim=8, train_windows=2).eval()
    x = torch.randint(0, 255, (2, 4, 12, 16, 16), dtype=torch.uint8)
    present = torch.tensor([[True, False, True, True], [False, True, False, False]])
    with torch.no_grad():
        dynamic, _ = model(x, present)
        model.static_shapes = True
        static, att = model(x, present)
    torch.testing.assert_close(static, dynamic, rtol=1e-4, atol=1e-5)      # same checkpoint, same outputs
    assert att.shape == (2, 4 * 4, 3)
    # Training: a fixed number of windows per slot, so the encoder always sees the same shape.
    seen = []
    model.enc.register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape)))
    model.train(); model.augment_in_model = False
    for _ in range(3):
        logits, _ = model(x, present)
        logits.sum().backward()
    assert seen == [(2 * 4 * 2, 3, 16, 16)] * 3
    # CPU jitter: shape and dtype kept, empty series stay empty, content actually moves.
    g = torch.Generator().manual_seed(1)
    study = x[0].clone(); study[1] = 0
    out = augment_series_cpu(study, g)
    assert out.shape == study.shape and out.dtype == torch.uint8
    assert int(out[1].max()) == 0 and not torch.equal(out[0], study[0])


def test_eval_window_subsets_encode_training_count(monkeypatch):
    # WINDOW_TTA: in eval, subset_windows encodes that many random windows per series (both paths);
    # None keeps the all-window pass bit-identical.
    from model_core import Reader25D
    patch_vit(monkeypatch)
    torch.manual_seed(0)
    model = Reader25D('cnn_tiny', 3, 4, pretrained=False, img=16, dim=8, train_windows=2).eval()
    x = torch.randint(0, 255, (2, 4, 12, 16, 16), dtype=torch.uint8)
    present = torch.tensor([[True, False, True, True], [True, True, True, True]])
    seen = []
    model.enc.register_forward_hook(lambda m, i, o: seen.append(int(i[0].shape[0])))
    with torch.no_grad():
        full, _ = model(x, present)
        model.subset_windows = 2
        sub, _ = model(x, present)
        model.static_shapes = True
        sub_static, _ = model(x, present)
        model.subset_windows, model.static_shapes = None, False
        again, _ = model(x, present)
    assert seen == [7 * 4, 7 * 2, 8 * 2, 7 * 4]          # present series x windows; static encodes every slot
    assert torch.equal(full, again) and not torch.allclose(full, sub)


def test_tpu_runtime_runs_the_reader_recipe_on_cpu(monkeypatch, tmp_path):
    # Without torch_xla the TPU runtime drives one CPU replica through the same loop; the
    # checkpoint must carry the GPU runtime's recipe so the submission notebook can rebuild it.
    import cv2, contextlib, re, gc, time, random
    from torch.utils.data import Dataset, DataLoader
    from tqdm.auto import tqdm
    from model_core import augment_series_cpu
    patch_vit(monkeypatch)
    n = 16; labels = ['a', 'b']; cache_dir = tmp_path / 'cache'; cache_dir.mkdir()
    rng = np.random.default_rng(3)
    np.save(cache_dir / 'cache.npy', rng.integers(0, 256, (n, 6, 24, 16, 16), dtype=np.uint8))
    present = {f'present_{i}': [True] * n if i < 3 else [i % 2 == 0] * n for i in range(6)}
    df = pd.DataFrame(dict(StudyInstanceUID=[f'id{i}' for i in range(n)], source=['pseudo'] * 12 + ['gold'] * 4,
                           row=range(n), cache_id=0, a=[.2, .9] * 8, b=[1, 0, 0, 1] * 4,
                           w_a=[1.] * 12 + [3.] * 4, w_b=[1.] * 12 + [3.] * 4,
                           report_group=[f'g{i // 2}' for i in range(n)], **present))
    meta = pp.preprocessing_meta('knee-2c-v3')
    out = tmp_path / 'reader_student'; out.mkdir()
    namespace = dict(os=os, gc=gc, copy=copy, math=__import__('math'), time=time, json=json, random=random, np=np,
                     pd=pd, torch=torch, nn=nn, F=torch.nn.functional, Dataset=Dataset, DataLoader=DataLoader, tqdm=tqdm,
                     cv2=cv2, df=df, CACHE_DIRS=[str(cache_dir)], CACHE_META=meta, PLANES=meta['slots'], LABEL_COLS=labels,
                     T_TRAIN=10, NUM_WORKERS=0, DEV='cpu', N_FOLDS=2, SEED=42, GROUP_COL='report_group', N_SLICES=24,
                     NORMALIZATION='pretrained', EPOCHS=2, FOLDS_TO_RUN=[0, 1], RESUME=False, GOLD_WEIGHT=3.,
                     TRAIN_WINDOWS=6, assign_folds=assign_folds, macro_auc=macro_auc, build_model=build_model,
                     slice_thirds=slice_thirds, window_centres=window_centres, augment_series_cpu=augment_series_cpu,
                     GRAD_CHECKPOINT=False, SLOT_DROPOUT=.2, TEACHER_SUMMARY={'weight': [.5, .9]}, write_atomic=write_atomic,
                     ARCH='reader25d', PRESET='reader_student', OUT=str(out), BACKBONE='cnn_tiny', IMG=16, BATCH=4,
                     ACCUM_STEPS=2, LR=5e-4, ENCODER_LR_SCALE=.2, WD=.02, PCT_START=.1, DROPOUT=.2, UNFREEZE_LAST=None,
                     HEAD_HIDDEN=8, EMA_DECAY=.9, CLIP=3., __name__='__main__')
    exec(cell('training.ipynb', 9), namespace)
    exec(cell('training.ipynb', 13), namespace)
    monkeypatch.setenv('KNEE_TPU_WORKERS', '0')
    exec((ROOT / 'tpu_runtime.py').read_text(), namespace)
    oof = pd.read_csv(out / 'oof_predictions.csv')
    assert len(oof) == 12 and oof[labels].notna().all().all()
    assert len(pd.read_csv(out / 'gold_fold_0.csv')) == 4
    saved = torch.load(out / 'fold_1.pt', weights_only=False)
    gpu_keys = set(re.findall(r'(\w+)=', (ROOT / 'train_runtime.py').read_text().split('recipe = dict(')[1].split('\nwrite_atomic')[0]))
    assert set(saved['cfg']) - {'trained_on'} == gpu_keys and saved['cfg']['trained_on'] == 'cpu'
    assert saved['cfg']['slot_dropout'] == .2 and saved['cfg']['preset'] == 'reader_student'
    # The saved EMA weights rebuild with the ordinary (dynamic) reader, as the submission does.
    model = build_model(saved['cfg'], len(labels), 6, pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    assert len(pd.read_csv(out / 'history_fold_0.csv')) == 2


def test_tpu_session_runner_runs_jobs_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv('KNEE_WORK', str(tmp_path))
    src = ("import os, json\n"
           "print('TPU runtime: CPU | folds', os.environ['KNEE_FOLDS'], flush=True)\n"
           "open(os.path.join(os.environ['KNEE_OUT'], 'done.txt'), 'w').write(os.environ.get('KNEE_SEED', ''))\n"
           "raise SystemExit(1 if os.environ['KNEE_PRESET'] == 'bad' else 0)\n")
    runner = (ROOT / 'tpu_session_runner.py').read_text()
    ns = dict(TPU_JOBS=[('reader_student2', [2, 3]), ('reader_student', [1], 7)], TPU_JOB_SOURCE=src)
    exec(runner, ns)
    assert (tmp_path / 'reader_student2' / 'done.txt').read_text() == ''
    assert (tmp_path / 'reader_student_s7' / 'done.txt').read_text() == '7'
    with pytest.raises(AssertionError, match='TPU jobs failed'):
        exec(runner, dict(TPU_JOBS=[('bad', [0]), ('reader_student', [0])], TPU_JOB_SOURCE=src))
    assert not (tmp_path / 'reader_student' / 'done.txt').exists()      # stops at the first failure


def test_report_grounded_student_session_trains_aux_head(tmp_path, monkeypatch):
    # Session V: the round-1 student with the report-embedding auxiliary loss. Training batches
    # carry the embedding after the labels; the checkpoint rebuilds (aux head included) for inference.
    rng = np.random.default_rng(5)
    ids = [f'1.2.{i}' for i in range(40)]
    def extra(root):
        rep = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f'rep_{i}' for i in range(4)])
        rep.insert(0, 'StudyInstanceUID', ids)
        rep.iloc[3, 1:] = np.nan                       # a study without an embedding is masked
        (root / 'repemb').mkdir(); rep.to_csv(root / 'repemb' / 'report_embeddings.csv', index=False)
        o = root / 'sessA' / 'reader_convnext'; o.mkdir(parents=True)
        oof = pd.DataFrame(rng.uniform(size=(34, 12)), columns=LABELS12)
        oof.insert(0, 'StudyInstanceUID', ids[:34]); oof['fold'] = 0
        oof.to_csv(o / 'oof_fold_0.csv', index=False)
    work, ns = run_session('04_session_V_student_rep_f0_f1.ipynb', tmp_path, monkeypatch, extra=extra)
    out = work / 'reader_student_rep'
    for fold in (0, 1):
        hist = pd.read_csv(out / f'history_fold_{fold}.csv')
        assert 'report_aux' in hist and np.isfinite(hist['report_aux']).all() and (hist['report_aux'] > 0).all()
    saved = torch.load(out / 'fold_0.pt', weights_only=False)
    assert saved['cfg']['aux_dim'] == 4 and saved['cfg']['report_aux']['dims'] == 4
    assert saved['cfg']['report_aux']['studies'] == 39 and saved['cfg']['preset'] == 'reader_student_rep'
    import timm
    fake = {}
    exec(SITECUSTOMIZE.replace('timm.create_model = ', '_create = '), fake)   # the sessions' tiny encoders
    monkeypatch.setattr(timm, 'create_model', fake['_create'])
    model = build_model(saved['cfg'], 12, 6, pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    assert model.aux is not None


def test_v4_student_session_trains_eight_slots(tmp_path, monkeypatch):
    # Session Z: the ConvNeXt student on a knee-2c-v4 cache (six v3 slots + two local slots).
    rng = np.random.default_rng(6)
    ids = [f'1.2.{i}' for i in range(40)]
    def extra(root):
        o = root / 'sessA' / 'reader_convnext'; o.mkdir(parents=True)
        oof = pd.DataFrame(rng.uniform(size=(34, 12)), columns=LABELS12)
        oof.insert(0, 'StudyInstanceUID', ids[:34]); oof['fold'] = 0
        oof.to_csv(o / 'oof_fold_0.csv', index=False)
    work, ns = run_session('04_session_Z_student_v4_f0_f1.ipynb', tmp_path, monkeypatch, extra=extra, version='knee-2c-v4')
    saved = torch.load(work / 'reader_student_v4' / 'fold_0.pt', weights_only=False)
    assert saved['cfg']['preprocessing']['version'] == 'knee-2c-v4' and len(saved['cfg']['planes']) == 8
    import timm
    fake = {}
    exec(SITECUSTOMIZE.replace('timm.create_model = ', '_create = '), fake)
    monkeypatch.setattr(timm, 'create_model', fake['_create'])
    model = build_model(saved['cfg'], 12, 8, pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    assert model.slot_emb.num_embeddings == 8


def test_v4_preset_refuses_a_v3_cache(tmp_path, monkeypatch):
    # Each job stops at its configuration check (before any training); the reason is in its log.
    with pytest.raises(AssertionError, match='Jobs failed'):
        run_session('04_session_Z_student_v4_f0_f1.ipynb', tmp_path, monkeypatch, version='knee-2c-v3')
    logs = ''.join(p.read_text() for p in (tmp_path / 'working').rglob('*.log'))
    assert 'Attach the knee-2c-v4 cache outputs' in logs


def test_v3_and_v4_members_share_one_v4_decode(tmp_path, monkeypatch):
    # A v3 member runs on the first six slots of v4 volumes; the generated submission check accepts the
    # pair, refuses a v3 member whose geometry differs, and both members predict every decoded study.
    import timm, glob, traceback
    fake = {}
    exec(SITECUSTOMIZE.replace('timm.create_model = ', '_create = '), fake)
    monkeypatch.setattr(timm, 'create_model', fake['_create'])
    labels = LABELS12
    v3, v4 = pp.preprocessing_meta('knee-2c-v3'), pp.preprocessing_meta('knee-2c-v4')
    paths = []
    for meta, preset in [(v4, 'reader_student_v4'), (v3, 'reader_student')]:
        cfg = dict(arch='reader25d', backbone='cnn_tiny', img=240, planes=meta['slots'], preprocessing=meta,
                   label_cols=labels, n_slices=24, head_hidden=8, drop=.2, train_windows=6, preset=preset)
        model = build_model(cfg, len(labels), len(meta['slots']), pretrained=False)
        path = tmp_path / preset / 'fold_0.pt'; path.parent.mkdir()
        torch.save(dict(model=model.state_dict(), cfg=cfg, fold=0), path); paths.append(str(path))
    nb = json.loads((ROOT / 'RUN_ON_KAGGLE' / '05i_submit_own_students.ipynb').read_text())
    config = next(''.join(c['source']) for c in nb['cells'] if 'SHARED = (' in ''.join(c['source']))
    check = config[config.index('SHARED = ('):config.index('families = {}')]
    cfg = torch.load(paths[0], weights_only=False)['cfg']
    exec(check, dict(torch=torch, CKPT_PATHS=paths, cfg=cfg, MODERN=True))
    bad = dict(torch.load(paths[1], weights_only=False)); bad['cfg'] = dict(bad['cfg'], preprocessing=dict(v3, crop_mm=120.0))
    torch.save(bad, tmp_path / 'bad.pt')
    with pytest.raises(AssertionError, match='different preprocessing'):
        exec(check, dict(torch=torch, CKPT_PATHS=[paths[0], str(tmp_path / 'bad.pt')], cfg=cfg, MODERN=True))
    comp = tmp_path / 'competition'; comp.mkdir()
    pd.DataFrame(dict(StudyInstanceUID=['s0', 's1'], **{c: [0, 0] for c in labels})).to_csv(comp / 'sample_submission.csv', index=False)
    manifest = pd.DataFrame(dict(StudyInstanceUID=['s0', 's1'], study_side=['L', 'L'],
                                 **{f'dir_{i}': ['d', 'd'] for i in range(8)}))
    rng = np.random.default_rng(0)
    namespace = dict(os=os, gc=__import__('gc'), time=__import__('time'), json=json, np=np, pd=pd, torch=torch,
                     Dataset=torch.utils.data.Dataset, tqdm=lambda x, **k: x, glob=glob, traceback=traceback,
                     manifest=manifest, MODERN=True, SLOTS=pp.RECIPES['knee-2c-v4']['slots'], RECIPE=pp.recipe('knee-2c-v4'),
                     slot_kwargs=pp.slot_kwargs, row_anchor=pp.row_anchor, cfg=cfg, CKPT_PATHS=paths, PLANES=v4['slots'],
                     LABEL_COLS=labels, N_SLICES=24, IMG=16, DEV='cpu', BATCH=2, COMP_DIR=str(comp), MAX_HOURS=1,
                     OUT_CSV=str(tmp_path / 'submission.csv'), build_model=build_model, rank_blend=rank_blend,
                     load_series=lambda *a, **k: rng.integers(0, 256, (24, 16, 16), dtype=np.uint8),
                     FAMILY_WEIGHTS={'reader_student': 1.0, 'reader_student_v4': 1.0})
    exec((ROOT / 'inference_runtime.py').read_text(), namespace)
    diagnostics = json.loads((tmp_path / 'inference_diagnostics.json').read_text())
    assert [m['predicted'] for m in diagnostics['models']] == [2, 2] and diagnostics['ensemble_members'] == 2


def test_reader_takes_v5_ten_windows(monkeypatch):
    # knee-2c-v5: 30 slices = 10 windows of 3 per series; dynamic and static (TPU) paths agree in eval.
    import timm
    fake = {}
    exec(SITECUSTOMIZE.replace('timm.create_model = ', '_create = '), fake)
    monkeypatch.setattr(timm, 'create_model', fake['_create'])
    meta = pp.preprocessing_meta('knee-2c-v5')
    cfg = dict(arch='reader25d', backbone='cnn_tiny', img=32, planes=meta['slots'], preprocessing=meta,
               head_hidden=16, drop=.2, train_windows=6)
    model = build_model(cfg, 12, 8, pretrained=False).eval()
    x = torch.randint(0, 255, (2, 8, 30, 32, 32), dtype=torch.uint8)
    present = torch.ones(2, 8, dtype=torch.bool); present[1, 3:6] = False
    with torch.no_grad():
        dyn = model(x, present)[0]
        model.static_shapes = True
        sta = model(x, present)[0]
    assert dyn.shape == (2, 12)
    torch.testing.assert_close(dyn, sta, atol=1e-4, rtol=1e-4)
