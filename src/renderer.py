import torch
import torch.nn.functional as F

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists: 1. - torch.exp(-raw * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = raw[..., :3]
    sigma = raw[..., 3]

    if raw_noise_std > 0.0:
        noise = torch.randn(sigma.shape, device=sigma.device) * raw_noise_std
        sigma = sigma + noise

    alpha = raw2alpha(sigma, dists)

    weights = alpha * torch.cumprod(torch.cat([
        torch.ones((alpha.shape[0], 1), device=alpha.device), 
        1. - alpha + 1e-10
    ], -1), -1)[..., :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, weights, depth_map, acc_map

def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0., 1., N_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def render_rays(rays_o, rays_d, model, kwargs):
    N_rays = rays_o.shape[0]
    device = rays_o.device
    
    N_samples = kwargs.get('N_samples', 64)      
    N_samples_hier = kwargs.get('N_samples_hier', 128) 
    near = kwargs.get('near', 2.0)
    far = kwargs.get('far', 6.0)
    
    t_vals = torch.linspace(0., 1., N_samples, device=device)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])
    
    if kwargs.get('perturb', True):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d[..., None, :].expand(-1, N_samples, -1).reshape(-1, 3)
    
    raw_coarse, density_coarse = model(pts_flat, dirs_flat)
        
    raw_coarse = raw_coarse.reshape(N_rays, N_samples, 3)
    density_coarse = density_coarse.reshape(N_rays, N_samples, 1)
    raw_all_coarse = torch.cat([raw_coarse, density_coarse], -1) 
    
    rgb_map_coarse, weights_coarse, _, _ = raw2outputs(
        raw_all_coarse, z_vals, rays_d, 
        raw_noise_std=kwargs.get('raw_noise_std', 0)
    )
    
    if N_samples_hier > 0:
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], N_samples_hier, det=not kwargs.get('perturb', True))
        z_samples = z_samples.detach() 
        
        z_vals_all = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)[0]
        
        pts_all = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_all[..., :, None]
        pts_all_flat = pts_all.reshape(-1, 3)
        dirs_all_flat = rays_d[..., None, :].expand(-1, pts_all.shape[1], -1).reshape(-1, 3)
        
        raw_fine, density_fine = model(pts_all_flat, dirs_all_flat)
        
        raw_fine = raw_fine.reshape(N_rays, pts_all.shape[1], 3)
        density_fine = density_fine.reshape(N_rays, pts_all.shape[1], 1)
        raw_all_fine = torch.cat([raw_fine, density_fine], -1)
        
        rgb_map, _, depth_map, acc_map = raw2outputs(
            raw_all_fine, z_vals_all, rays_d,
            raw_noise_std=kwargs.get('raw_noise_std', 0)
        )
    else:
        rgb_map = rgb_map_coarse
        depth_map = None
        acc_map = None
        
    return {
        'rgb_map': rgb_map,
        'rgb_map_coarse': rgb_map_coarse,
        'depth_map': depth_map,
        'acc_map': acc_map
    }

def render_batch(rays_o, rays_d, model, chunk=1024*4, **kwargs):
    all_ret = {}
    for i in range(0, rays_o.shape[0], chunk):
        ret = render_rays(rays_o[i:i+chunk], rays_d[i:i+chunk], model, kwargs)
        
        for k in ret:
            if ret[k] is not None:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
            
    final_ret = {}
    for k in all_ret:
        if len(all_ret[k]) > 0:
            final_ret[k] = torch.cat(all_ret[k], 0)
        else:
            final_ret[k] = None
            
    return final_ret