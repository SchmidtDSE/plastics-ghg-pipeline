"""Data structures for model observations and predictions.

License: BSD
"""

import typing


class Observation:
    """An individual observation from the dataset of net trade."""

    def __init__(self, ratio: float, gdp: float, population: float):
        """Create a new observation record.

        Args:
            ratio: The ratio of sector net trade to overall net trade for a region.
            gdp: The estimated GDP for the region for the year of this observation.
            population: The estimated population for the region for the year of this observation.
        """
        self._ratio = ratio
        self._gdp = gdp
        self._population = population

    def get_ratio(self) -> float:
        """Get the sector trade ratio.

        Returns:
            The ratio of sector net trade to overall net trade for a region.
        """
        return self._ratio

    def get_gdp(self) -> float:
        """Get the estimated GDP associated with this observation.

        Returns:
            The estimated GDP for the region for the year of this observation.
        """
        return self._gdp

    def get_population(self) -> float:
        """Get the estimated population associated with this observation.

        Returns:
            The estimated population for the region for the year of this observation.
        """
        return self._population

    def to_dict(self) -> typing.Dict:
        """Serialize this record to a dictionary containing only primitives.

        Returns:
            Dictionary serialization of this record.
        """
        return {
            'ratioSector': self._ratio,
            'gdp': self._gdp,
            'population': self._population
        }

    @classmethod
    def from_dict(cls, target: typing.Dict) -> 'Observation':
        """Deserialize this record from a dictionary containing only primitives.

        Returns:
            Parsed version of this record.
        """
        return Observation(
            float(target['ratioSector']),
            float(target['gdp']),
            float(target['population'])
        )


class Change:
    """Record of a delta between years in input variables and response."""

    def __init__(self, region: str, sector: str, year: int, years: int, gdp_change: float,
        population_change: float, before_ratio: float, after_ratio: float):
        """Create a new record describing a change between years.

        Args:
            region: The region like "NAFTA" in which the change is observed or predicted. Case
                insensitive.
            sector: The sector like "Packaging" in which the change is observed or predicted.
            year: The base year for the "before" values. Case insensitive.
            years: The time gap in years between the start and end year. May be negative.
            gdp_change: The percent change in GDP in the region where 1% is 0.01.
            population_change: The percent change in population in the region where 1% is 0.01.
            before_ratio: The prior ratio of sector to overall net trade in the region (in year).
            after_ratop: The after ratio of sector to overall net trade in the region (in year +
                years).
        """
        self._region = region.lower()
        self._sector = sector.lower()
        self._year = year
        self._years = years
        self._gdp_change = gdp_change
        self._population_change = population_change
        self._before_ratio = before_ratio
        self._after_ratio = after_ratio

    def get_region(self) -> str:
        """Get the region that this observation, estimate, or prediction is for.

        Returns:
            The region like "NAFTA" in which the change is observed or predicted.
        """
        return self._region

    def get_sector(self) -> str:
        """Get the sector that this observation, estimate, or prediction is for.

        Returns:
            The sector like "Packaging" in which the change is observed or predicted.
        """
        return self._sector

    def get_year(self) -> int:
        """Get the starting year that this observation, estimate, or prediction is for.

        Returns:
            The base year for the "before" values.
        """
        return self._year

    def get_years(self) -> int:
        """Get the year delta for this observation, estimate, or prediction.

        Returns:
            The time gap in years between the start and end year. May be negative.
        """
        return self._years

    def get_gdp_change(self) -> float:
        """Get the change in gross domestic product for this record's region.

        Returns:
            The percent change in GDP in the region where 1% is 0.01.
        """
        return self._gdp_change

    def get_population_change(self) -> float:
        """Get the change in population for this record's region.

        Returns:
            The percent change in population in the region where 1% is 0.01.
        """
        return self._population_change

    def get_before_ratio(self) -> float:
        """Get the ratio of sector to overall net trade in the "before" year.

        Returns:
            The prior ratio of sector to overall net trade in the region (in year).
        """
        return self._before_ratio

    def get_after_ratio(self) -> float:
        """Get the ratio of sector to overall net trade in the "after" year.

        Returns:
            The after ratio of sector to overall net trade in the region (in year + years).
        """
        return self._after_ratio

    def to_dict(self) -> typing.Dict:
        """Serialize this record to a dictionary containing only primitives.

        Returns:
            Dictionary serialization of this record.
        """
        return {
            'region': self._region,
            'sector': self._sector,
            'year': self._year,
            'years': self._years,
            'gdpChange': self._gdp_change,
            'populationChange': self._population_change,
            'beforeRatio': self._before_ratio,
            'afterRatio': self._after_ratio,
        }

    def to_vector(self) -> typing.Tuple[typing.Union[int, float]]:
        """Return a vectorization of this record usable in machine learning.

        Returns:
            Vector version of this record.
        """
        return (
            self._years,
            self._gdp_change,
            self._population_change,
            self._before_ratio,
            self._hot_encode(self._region, 'China'),
            self._hot_encode(self._region, 'EU30'),
            self._hot_encode(self._region, 'NAFTA'),
            self._hot_encode(self._region, 'RoW'),
            self._hot_encode(self._sector, 'Agriculture'),
            self._hot_encode(self._sector, 'Building_Construction'),
            self._hot_encode(self._sector, 'Electrical_Electronic'),
            self._hot_encode(self._sector, 'Household_Leisure_Sports'),
            self._hot_encode(self._sector, 'Others'),
            self._hot_encode(self._sector, 'Packaging'),
            self._hot_encode(self._sector, 'Textiles'),
            self._hot_encode(self._sector, 'Transportation')
        )

    @classmethod
    def from_dict(cls, target: typing.Dict) -> 'Change':
        """Deserialize this record from a dictionary containing only primitives.

        Returns:
            Parsed version of this record.
        """
        return Change(
            str(target['region']),
            str(target['sector']),
            int(target['year']),
            int(target['years']),
            float(target['gdpChange']),
            float(target['populationChange']),
            float(target['beforeRatio']),
            float(target['afterRatio'])
        )

    def _hot_encode(self, candidate: str, target: str) -> int:
        """Convienence function for one-hot encoding.

        Args:
            candidate: The value observed in the record.
            target: The value for which the current vector element should be 1 if matching and 0
                otherwise.

        Returns:
            1 if candidate and target are the same (case insensitive) and 0 otherwise.
        """
        return 1 if candidate.lower() == target.lower() else 0


class ObservationIndex:
    """Collection of indexed observations for fast lookup."""

    def __init__(self):
        """Create a new empty collection of observations."""
        self._records: typing.Dict[str, Observation] = {}
        self._years: typing.Set[int] = set()
        self._regions: typing.Set[str] = set()
        self._sectors: typing.Set[str] = set()

    def add(self, year: int, region: str, sector: str, record: Observation):
        """Add a new record into this index.

        Args:
            year: The year for which the observation was estimated or in which it was actually made.
            region: The region like "China" in which the observation was made or for which it was
                estimated. Case insensitive.
            sector: The sector like "Packaging" in which the observation was made or for which it
                was estimated. Case insensitive.
            record: The record to add to the index.
        """
        region_lower = region.lower()
        sector_lower = sector.lower()
        self._years.add(year)
        self._regions.add(region_lower)
        self._sectors.add(sector_lower)

        key = self._get_key(year, region_lower, sector_lower)

        self._records[key] = record

    def get_record(self, year: int, region: str, sector: str) -> typing.Optional[Observation]:
        """Lookup a record.

        Args:
            year: The desired year for the observation.
            region: The desired region like "RoW" for the observation.
            sector: The desired sector like "Agriculture" for the observation.

        Returns:
            The observation if one is found in the index and None otherwise.
        """
        key = self._get_key(year, region, sector)
        return self._records.get(key, None)

    def get_change(self, year: int, region: str, sector: str,
        years: int) -> typing.Optional[Change]:
        """Calculate estimated or actual change between two years within a region and sector.

        Args:
            year: The base or starting year desired.
            region: The region in which to get the change like "EU30".
            sector: The sector like "Building_Construction" in which to get the change.
            years: The delta from the base or starting year to the end year such that end year is
                year + years.

        Returns:
            Change over the specified time range estimated or expected, returning None if data are
            not available.
        """
        before = self.get_record(year, region, sector)
        after = self.get_record(year + years, region, sector)

        if before is None or after is None:
            return None

        return Change(
            region,
            sector,
            year,
            years,
            self._calculate_change(before.get_gdp(), after.get_gdp()),
            self._calculate_change(before.get_population(), after.get_population()),
            before.get_ratio(),
            after.get_ratio()
        )

    def get_years(self) -> typing.Iterable[int]:
        """Get all observed years.

        Returns:
            Iterable over the years seen in this index.
        """
        return self._years

    def has_year(self, target: int) -> bool:
        """Determine if this index contains the given year.

        Args:
            target: The value to check for.

        Returns:
            True if found and false otherwise.
        """
        return target in self._years

    def get_regions(self) -> typing.Iterable[str]:
        """Get all observed regions.

        Returns:
            Iterable over the regions seen in this index.
        """
        return self._regions

    def has_region(self, target: int) -> bool:
        """Determine if this index contains the given region.

        Args:
            target: The value to check for.

        Returns:
            True if found and false otherwise.
        """
        return target.lower() in self._regions

    def get_sectors(self) -> typing.Iterable[str]:
        """Get all observed sectors.

        Returns:
            Iterable over the sectors seen in this index.
        """
        return self._sectors

    def has_sector(self, target: int) -> bool:
        """Determine if this index contains the given sector.

        Args:
            target: The value to check for.

        Returns:
            True if found and false otherwise.
        """
        return target.lower() in self._sectors

    def _get_key(self, year: int, region: str, sector: str) -> str:
        """Get a string which uniquely identifies a year, region, sector combination.

        Args:
            year: Desired year.
            region: Desired region.
            sector: Desired sector.

        Returns:
            Unique string (uniqueness enforced case insensitive).
        """
        pieces = [year, region, sector]
        pieces_str = [str(x) for x in pieces]

        pieces_invalid = filter(lambda x: '\t' in x, pieces_str)
        pieces_invalid_count = sum(map(lambda x: 1, pieces_invalid))
        if pieces_invalid_count > 0:
            raise RuntimeError('Tabs not allowed in region or sector names.')

        return '\t'.join(pieces_str).lower()

    def _calculate_change(self, before: float, after: float) -> float:
        """Calculate the change between two values.

        Args:
            before: The prior value.
            after: The new value.

        Returns:
            Change where 0.01 refers to 1%.
        """
        if before == 0:
            return 0
        else:
            return (after - before) / before
