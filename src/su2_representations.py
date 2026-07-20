from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, order=True, slots=True)
class Irrep:
    '''SU(2) irrep stored exactly as j2 = 2j.'''

    j2: int

    def __post_init__(self) -> None:
        if not isinstance(self.j2, int) or isinstance(self.j2, bool) or self.j2 < 0:
            raise ValueError('j2 must be a nonnegative integer.')

    @property
    def dimension(self) -> int:
        return self.j2 + 1

    @property
    def casimir(self) -> Fraction:
        return Fraction(self.j2 * (self.j2 + 2), 4)

    @property
    def spin(self) -> Fraction:
        return Fraction(self.j2, 2)

    def dual(self) -> 'Irrep':
        return self

    def reverse_orientation(self) -> 'Irrep':
        return self.dual()

    def tensor_product(self, other: 'Irrep') -> tuple['Irrep', ...]:
        if not isinstance(other, Irrep):
            raise TypeError('SU(2) tensor product requires another Irrep.')
        return tuple(Irrep(j2) for j2 in range(abs(self.j2 - other.j2), self.j2 + other.j2 + 1, 2))


CONVENTION = {
    'irrep_coordinate': 'j2=2j nonnegative integer',
    'dimension': 'd_j=j2+1',
    'casimir': 'C2=j2(j2+2)/4=j(j+1)',
    'dual': 'SU(2) irreps are self-dual',
    'class_angle': 'Tr(U)=2 cos(theta)',
    'normalized_weight': 'exp(beta*(cos(theta)-1))',
}
