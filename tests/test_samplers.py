import copy
import math
import pytest
import torch
from cedr.generation import grid, sample, sample_ode
from cedr.samplers import edm_plan, noise_seed, quantile
from cedr.models import build_models, make_optimizers
from cedr.train import stage_update_count


class Probe:
    def __init__(self):
        self.inputs = []
        self.predictions = []
    def __call__(self, x, t, **kwargs):
        self.inputs.append((x.clone(), tuple(v.clone() for v in t)))
        z, sc = x.chunk(2, -1)
        pred = 0.25 * z + 0.1 * sc + 0.5
        self.predictions.append(pred.clone())
        return pred, None


@pytest.mark.parametrize("method", ["dmd2", "edm"])
@pytest.mark.parametrize("steps", [1, 2, 5])
def test_stochastic_masks_counts_terminal_and_rng(cfg, method, steps):
    condition = torch.randn(2, 6, 12)
    mask = torch.zeros(2, 6, dtype=torch.bool); mask[:, :2] = True
    probe = Probe(); gen = torch.Generator().manual_seed(42)
    z, counts = sample(probe, condition, mask, cfg, generator=gen, sampler=method, steps=steps, batch_offset=32)
    assert torch.equal(z[mask], condition[mask])
    assert torch.equal(z[~mask], probe.predictions[-1][~mask])
    assert counts['denoiser_calls'] == steps
    assert counts['noise_draws'] == steps - 1
    assert counts['applied_noise_intervals'] == (steps - 1 if method == 'dmd2' else max(steps - 2, 0))
    assert counts['noise_seed'] == noise_seed(42, 32)
    control = torch.Generator().manual_seed(42)
    torch.randn(condition.shape, generator=control)
    assert torch.equal(control.get_state(), gen.get_state())
    for inp, _ in probe.inputs:
        assert torch.equal(inp[..., :12][mask], condition[mask])
        assert torch.equal(inp[..., 12:][mask], condition[mask])
    repeat, _ = sample(Probe(), condition, mask, cfg, generator=torch.Generator().manual_seed(42), sampler=method, steps=steps, batch_offset=32)
    assert torch.equal(z, repeat)


def test_dmd2_first_step_formula(cfg):
    condition = torch.zeros(1, 3, 12); mask = torch.zeros(1, 3, dtype=torch.bool)
    probe = Probe(); initial = torch.randn(condition.shape)
    sample(probe, condition, mask, cfg, generator=torch.Generator().manual_seed(7), sampler='dmd2', steps=3, initial_noise=initial)
    noise = torch.randn(condition.shape, generator=torch.Generator().manual_seed(noise_seed(7, 0))) * 2
    t = grid(3)[1]
    expected = torch.cat([t.pow(p) * clean + (1-t.pow(p)) * eps for p,clean,eps in zip(cfg['inference']['clock_powers'],probe.predictions[0].split(4,-1),noise.split(4,-1))],-1)
    assert torch.equal(probe.inputs[1][0][...,:12], expected)


def test_edm_index_churn_and_variance():
    times = grid(8).tolist(); powers = [2.5,2,1.5]
    plans = edm_plan(times,powers,.7,2,-1.5,.8)
    assert plans[2]['evaluation_tau'] == quantile(.7*2/8,-1.5,.8)
    assert plans[2]['evaluation_tau'] != .7*times[2]
    for i,plan in enumerate(plans):
        assert not plan['noise_applied'] if i in (0,7) else plan['noise_applied']
        for p,a,v in zip(powers,plan['alpha'],plan['variance']):
            t=times[i]**p; b=plan['evaluation_tau']**p
            assert math.isclose((2*a*(1-t))**2+v, (2*(1-b))**2,abs_tol=1e-12)
    assert not any(p['noise_applied'] for p in edm_plan(times,powers,1,2,-1.5,.8))
    for c in (-1,1.1,float('nan')):
        with pytest.raises(ValueError):edm_plan(times,powers,c,2,-1.5,.8)


def test_ode_fallback_and_explicit_selection(cfg):
    cfg=copy.deepcopy(cfg);cfg['inference'].pop('sampler',None)
    cond=torch.randn(2,4,12);mask=torch.zeros(2,4,dtype=torch.bool)
    a,_=sample(Probe(),cond,mask,cfg,generator=torch.Generator().manual_seed(42),steps=4)
    b,_=sample_ode(Probe(),cond,mask,cfg,generator=torch.Generator().manual_seed(42),steps=4)
    assert torch.equal(a,b)
    cfg['inference']['sampler']='dmd2'
    c,_=sample(Probe(),cond,mask,cfg,generator=torch.Generator().manual_seed(42),steps=4,sampler='ode')
    assert torch.equal(a,c)


def test_independent_prompt_learning_rates(cfg):
    elf,prompt=build_models(cfg,'cpu',torch.randn(64,16))
    opts=make_optimizers(elf,prompt,.002,prompt_lr=.0005)
    assert all(g['lr']==.002 for o in opts['elf'] for g in o.param_groups)
    assert all(g['lr']==.0005 for o in opts['prompt'] for g in o.param_groups)
    assert all(not o.state for values in opts.values() for o in values)
    defaults=make_optimizers(elf,prompt,.002)
    assert all(g['lr']==.002 for values in defaults.values() for o in values for g in o.param_groups)


def test_cumulative_epoch_rounding():
    assert stage_update_count(12,38036,8,policy='ceil_update')==57054
    assert stage_update_count(21,38036,8,policy='ceil_update')==99845
    assert stage_update_count(9,38036,8,policy='ceil_update',preceding_epochs=12)==42791
    assert stage_update_count(2,38036,8,policy='ceil_update')==9509
    with pytest.raises(ValueError):stage_update_count(9,38036,8)


def test_coding_epoch_tail_and_resume():
    from cedr.data import Cursor
    from cedr.train import stage_update_count
    for world in (2, 4, 8):
        cursor = Cursor(4893784, 32, world, 0, 42, drop_multiple=512)
        assert cursor.usable == 4893696
        assert stage_update_count(2, cursor.usable // (32 * world), 512 // (32 * world)) == 19116
        cursor.offset = cursor.usable - 32 * world
        cursor.next()
        saved = cursor.state_dict()
        restored = Cursor(4893784, 32, world, 0, 42, drop_multiple=512)
        restored.load_state_dict(saved)
        assert cursor.next() == restored.next()
        assert cursor.epoch == restored.epoch == 1
    prior = Cursor(65, 2, 2, 0, 42)
    state = prior.state_dict(); state.pop("drop_multiple")
    prior.load_state_dict(state)
