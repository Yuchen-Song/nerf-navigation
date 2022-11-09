import torch

def density_to_pc(density_fn, threshold=1e-1, N=1000000, window=None):
    #Converts density to a point cloud queried on a grid
    s = int(N**(1/3))   #s is number of points per side
    grid = torch.stack(torch.meshgrid(torch.linspace(window[0, 0], window[0, 1], s),
                                            torch.linspace(window[1, 0], window[1, 1], s),
                                            torch.linspace(window[2, 0], window[2, 1],  s)), dim=-1)

    grid = grid.reshape(-1, 3)
    densities = density_fn(grid)
    grid = grid[densities > threshold]

    return grid
