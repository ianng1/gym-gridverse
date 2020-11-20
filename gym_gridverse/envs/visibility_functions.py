import math
from functools import lru_cache
from typing import Iterator, List, Optional

import more_itertools as mitt
import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.geometry import Area, Position
from gym_gridverse.info import Grid
from gym_gridverse.rng import get_gv_rng_if_none


class VisibilityFunction(Protocol):
    def __call__(
        self,
        grid: Grid,
        position: Position,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> np.array:
        ...


def full_visibility(
    grid: Grid,
    position: Position,  # pylint: disable = unused-argument
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable = unused-argument
) -> np.ndarray:

    return np.ones((grid.height, grid.width), dtype=bool)


def minigrid_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable = unused-argument
) -> np.ndarray:

    if position.y != grid.height - 1:
        #  gym-minigrid does not handle this case, and we are not currently
        #  generalizing it
        raise NotImplementedError

    visibility = np.zeros((grid.height, grid.width), dtype=bool)
    visibility[position.y, position.x] = True  # agent

    for y in range(grid.height - 1, -1, -1):
        for x in range(grid.width - 1):
            if visibility[y, x] and grid[y, x].transparent:
                visibility[y, x + 1] = True
                if y > 0:
                    visibility[y - 1, x] = True
                    visibility[y - 1, x + 1] = True

        for x in range(grid.width - 1, 0, -1):
            if visibility[y, x] and grid[y, x].transparent:
                visibility[y, x - 1] = True
                if y > 0:
                    visibility[y - 1, x] = True
                    visibility[y - 1, x - 1] = True

    return visibility


# TODO test this
def ray_positions(
    start_pos: Position, area: Area, *, radians: float, step_size: float
) -> Iterator[Position]:
    y, x = float(start_pos.y), float(start_pos.x)
    dy = -math.sin(radians)
    dx = math.cos(radians)

    pos = start_pos
    while area.contains(pos):
        yield pos

        y += step_size * dy
        x += step_size * dx
        pos = Position(round(y), round(x))


# TODO test this
@lru_cache()
def rays_positions(start_pos: Position, area: Area) -> List[List[Position]]:
    rays: List[List[Position]] = []

    for degrees in range(360):
        # conversion to radians
        radians = degrees * math.pi / 180.0
        ray = ray_positions(start_pos, area, radians=radians, step_size=0.01)
        ray = mitt.unique_justseen(ray)
        rays.append(list(ray))

    return rays


def raytracing_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> np.ndarray:

    area = Area((0, grid.height - 1), (0, grid.width - 1))
    rays = rays_positions(position, area)

    counts_num = np.zeros((area.height, area.width), dtype=int)
    counts_den = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_num[pos.y, pos.x] += 1

            counts_den[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    # TODO add as parameter to function
    visibility = counts_num > 0  # at least one ray makes it
    # visibility = counts_num > 0.5 * counts_den # half of the rays make it
    # visibility = counts_num > 0.1 * counts_den  # 10% of the rays make it
    # visibility = counts_num > 1  # at least 2 rays make it

    return visibility


def stochastic_raytracing_visibility(  # TODO add test
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,
) -> np.ndarray:
    rng = get_gv_rng_if_none(rng)

    area = Area((0, grid.height - 1), (0, grid.width - 1))
    rays = rays_positions(position, area)

    counts_num = np.zeros((area.height, area.width), dtype=int)
    counts_den = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_num[pos.y, pos.x] += 1

            counts_den[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    probs = np.nan_to_num(counts_num / counts_den)
    visibility = probs <= rng.random(probs.shape)
    return visibility


def factory(name: str) -> VisibilityFunction:

    if name == 'full_visibility':
        return full_visibility

    if name == 'minigrid_visibility':
        return minigrid_visibility

    if name == 'raytracing_visibility':
        return raytracing_visibility

    if name == 'stochastic_raytracing_visibility':
        return stochastic_raytracing_visibility

    raise ValueError(f'invalid visibility function name {name}')
