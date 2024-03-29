"""Data structures for model observations and predictions.

License: BSD
"""
import csv
import typing

import const


def parse_ratio_str(target) -> typing.Optional[float]:
    """Parse string describing a ratio of subtype to overall trade or indicating that it is unknown.

    Args:
        target: The raw value to interpret.

    Returns:
        The ratio found from the input raw value or None if the value indicates that the ratio is
        not known.
    """
    ratio_str = str(target).strip().lower()
    no_ratio = ratio_str in const.RATIO_NONE_STRS
    return None if no_ratio else float(target)


class Observation:
    """An individual observation from the dataset of net trade."""

    def __init__(self, ratio: typing.Optional[float], gdp: float, population: float):
        """Create a new observation record.

        Args:
            ratio: The ratio of subtype net trade to overall net trade for a region if known or None
                if unknown.
            gdp: The estimated GDP for the region for the year of this observation.
            population: The estimated population for the region for the year of this observation.
        """
        self._ratio = ratio
        self._gdp = gdp
        self._population = population

    def get_ratio(self) -> typing.Optional[float]:
        """Get the subtype trade ratio.

        Returns:
            The ratio of subtype net trade to overall net trade for a region.
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
            'ratioSubtype': self.get_ratio(),
            'gdp': self.get_gdp(),
            'population': self.get_population()
        }

    @classmethod
    def from_dict(cls, target: typing.Dict) -> 'Observation':
        """Deserialize this record from a dictionary containing only primitives.

        Returns:
            Parsed version of this record.
        """
        return Observation(
            parse_ratio_str(target['ratioSubtype']),
            float(target['gdp']),
            float(target['population'])
        )


class Change:
    """Record of a delta between years in input variables and response."""

    def __init__(self, region: str, subtype: str, year: int, years: int, gdp_change: float,
        population_change: float, before_ratio: float, after_ratio: typing.Optional[float]):
        """Create a new record describing a change between years.

        Args:
            region: The region like "NAFTA" in which the change is observed or predicted. Case
                insensitive.
            subtype: The subtype like "Packaging" in which the change is observed or predicted.
            year: The base year for the "before" values. Case insensitive.
            years: The time gap in years between the start and end year. May be negative.
            gdp_change: The percent change in GDP in the region where 1% is 0.01.
            population_change: The percent change in population in the region where 1% is 0.01.
            before_ratio: The prior ratio of subtype to overall net trade in the region (in year).
            after_rato: The after ratio of subtype to overall net trade in the region (in year +
                years). Pass None if unknown.
        """
        self._region = region.lower()
        self._subtype = subtype.lower()
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

    def get_subtype(self) -> str:
        """Get the subtype that this observation, estimate, or prediction is for.

        Returns:
            The subtype like "Packaging" in which the change is observed or predicted.
        """
        return self._subtype

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
        """Get the ratio of subtype to overall net trade in the "before" year.

        Returns:
            The prior ratio of subtype to overall net trade in the region (in year).
        """
        return self._before_ratio

    def get_after_ratio(self) -> typing.Optional[float]:
        """Get the ratio of subtype to overall net trade in the "after" year.

        Returns:
            The after ratio of subtype to overall net trade in the region (in year + years).
        """
        return self._after_ratio

    def to_dict(self) -> typing.Dict:
        """Serialize this record to a dictionary containing only primitives.

        Returns:
            Dictionary serialization of this record.
        """
        return {
            'region': self.get_region(),
            'subtype': self.get_subtype(),
            'year': self.get_year(),
            'years': self.get_years(),
            'gdpChange': self.get_gdp_change(),
            'populationChange': self.get_population_change(),
            'beforeRatio': self.get_before_ratio(),
            'afterRatio': self.get_after_ratio(),
        }

    def to_vector(self) -> typing.Iterable[typing.Union[int, float]]:
        """Return a vectorization of this record usable in machine learning.

        Returns:
            Vector version of this record.
        """
        pieces_common: typing.List[typing.Union[int, float]] = [
            self.get_years(),
            self.get_gdp_change(),
            self.get_population_change(),
            self.get_before_ratio()
        ]
        pieces_region: typing.List[typing.Union[int, float]] = [
            self._hot_encode(self._region, x) for x in const.REGIONS
        ]
        pieces_subtype: typing.List[typing.Union[int, float]] = [
            self._hot_encode(self._subtype, x) for x in const.SUBTYPES
        ]
        return pieces_common + pieces_region + pieces_subtype

    def get_response(self) -> float:
        """Get the value to be predicted or that was predicted by the model.

        Returns:
            The after ratio of subtype to overall net trade in the region (in year + years).
        """
        response = self.get_after_ratio()

        if response is None:
            raise RuntimeError('Cannot provide response for unpredicted instance.')

        return response

    def has_response(self) -> bool:
        """Determine if the response variable value is available.

        Return:
            True if avilable and false otherwise.
        """
        return self.get_after_ratio() is not None

    @classmethod
    def from_dict(cls, target: typing.Dict) -> 'Change':
        """Deserialize this record from a dictionary containing only primitives.

        Returns:
            Parsed version of this record.
        """
        return Change(
            str(target['region']),
            str(target['subtype']),
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


class IndexedObservations:

    def add(self, year: int, region: str, subtype: str, record: Observation):
        """Add a new record into this index, overwriting if a matching record is already present.

        Args:
            year: The year for which the observation was estimated or in which it was actually made.
            region: The region like "China" in which the observation was made or for which it was
                estimated. Case insensitive.
            subtype: The subtype like "Packaging" in which the observation was made or for which it
                was estimated. Case insensitive.
            record: The record to add to the index.
        """
        raise NotImplementedError('Use implementor.')

    def get_record(self, year: int, region: str, subtype: str) -> typing.Optional[Observation]:
        """Lookup a record.

        Args:
            year: The desired year for the observation.
            region: The desired region like "RoW" for the observation.
            subtype: The desired subtype like "Agriculture" for the observation.

        Returns:
            The observation if one is found in the index and None otherwise.
        """
        raise NotImplementedError('Use implementor.')

    def get_change(self, year: int, region: str, subtype: str,
        years: int) -> typing.Optional[Change]:
        """Calculate estimated or actual change between two years within a region and subtype.

        Args:
            year: The base or starting year desired.
            region: The region in which to get the change like "EU30".
            subtype: The subtype like "Building_Construction" in which to get the change.
            years: The delta from the base or starting year to the end year such that end year is
                year + years.

        Returns:
            Change over the specified time range estimated or expected, returning None if data are
            not available.
        """
        raise NotImplementedError('Use implementor.')

    def get_years(self) -> typing.Iterable[int]:
        """Get all observed years.

        Returns:
            Iterable over the years seen in this index.
        """
        raise NotImplementedError('Use implementor.')

    def has_year(self, target: int) -> bool:
        """Determine if this index contains the given year.

        Args:
            target: The value to check for.

        Returns:
            True if found and false otherwise.
        """
        raise NotImplementedError('Use implementor.')

    def get_regions(self) -> typing.Iterable[str]:
        """Get all observed regions.

        Returns:
            Iterable over the regions seen in this index.
        """
        raise NotImplementedError('Use implementor.')

    def has_region(self, target: str) -> bool:
        """Determine if this index contains the given region.

        Args:
            target: The value to check for.

        Returns:
            True if found and false otherwise.
        """
        raise NotImplementedError('Use implementor.')

    def get_subtypes(self) -> typing.Iterable[str]:
        """Get all observed subtypes.

        Returns:
            Iterable over the subtypes seen in this index.
        """
        raise NotImplementedError('Use implementor.')

    def has_subtype(self, target: str) -> bool:
        """Determine if this index contains the given subtype.

        Args:
            target: The value to check for.

        Returns:
            True if found and false otherwise.
        """
        raise NotImplementedError('Use implementor.')

    def _calculate_change(self, before: float, after: float) -> float:
        """Calculate the change between two values.

        Args:
            before: The prior value.
            after: The new value.

        Returns:
            Change where 0.01 refers to 1%.
        """
        if before == 0:
            raise RuntimeError('Before is zero, causing divison by zero.')

        return (after - before) / before


class KeyingObservationIndex(IndexedObservations):
    """Collection of indexed observations for fast lookup."""

    def __init__(self):
        """Create a new empty collection of observations."""
        self._records: typing.Dict[str, Observation] = {}
        self._years: typing.Set[int] = set()
        self._regions: typing.Set[str] = set()
        self._subtypes: typing.Set[str] = set()

    def add(self, year: int, region: str, subtype: str, record: Observation):
        region_lower = region.lower()
        subtype_lower = subtype.lower()
        self._years.add(year)
        self._regions.add(region_lower)
        self._subtypes.add(subtype_lower)

        key = self._get_key(year, region_lower, subtype_lower)

        self._records[key] = record

    def get_record(self, year: int, region: str, subtype: str) -> typing.Optional[Observation]:
        key = self._get_key(year, region, subtype)
        return self._records.get(key, None)

    def get_change(self, year: int, region: str, subtype: str,
        years: int) -> typing.Optional[Change]:
        before = self.get_record(year, region, subtype)
        after = self.get_record(year + years, region, subtype)

        if before is None or after is None:
            return None

        before_ratio = before.get_ratio()
        after_ratio = after.get_ratio()
        if before_ratio is None or after_ratio is None:
            return None

        return Change(
            region,
            subtype,
            year,
            years,
            self._calculate_change(before.get_gdp(), after.get_gdp()),
            self._calculate_change(before.get_population(), after.get_population()),
            before_ratio,
            after_ratio
        )

    def get_years(self) -> typing.Iterable[int]:
        return self._years

    def has_year(self, target: int) -> bool:
        return target in self._years

    def get_regions(self) -> typing.Iterable[str]:
        return self._regions

    def has_region(self, target: str) -> bool:
        return target.lower() in self._regions

    def get_subtypes(self) -> typing.Iterable[str]:
        return self._subtypes

    def has_subtype(self, target: str) -> bool:
        return target.lower() in self._subtypes

    def _get_key(self, year: int, region: str, subtype: str) -> str:
        """Get a string which uniquely identifies a year, region, subtype combination.

        Args:
            year: Desired year.
            region: Desired region.
            subtype: Desired subtype.

        Returns:
            Unique string (uniqueness enforced case insensitive).
        """
        pieces = [year, region, subtype]
        pieces_str = [str(x) for x in pieces]

        pieces_invalid = filter(lambda x: '\t' in x, pieces_str)
        pieces_invalid_count = sum(map(lambda x: 1, pieces_invalid))
        if pieces_invalid_count > 0:
            raise RuntimeError('Tabs not allowed in region or subtype names.')

        return '\t'.join(pieces_str).lower()


def get_observation_included(require_response: bool, record: Observation) -> bool:
    """Determine if an observation should be included in a dataset.

    Args:
        require_response: Flag indicating if known ratios are required. True if the ratio of subtype
            to overall trade needs to be known for the observation to be in the dataset and False if
            it is not required.
        record: The record to consider.

    Returns:
        True if the record should be included in the dataset and false otherwise.
    """
    return (not require_response) or (record.get_ratio() is not None)


def build_index_from_file(path: str, require_response: bool = False) -> IndexedObservations:
    """Build an indexed dataset of observations.

    Args:
        path: The location of the CSV file from which the index should be constructed.
        require_response: Flag indicating if known ratios are required. True if the ratio of subtype
            to overall trade needs to be known for the observation to be in the dataset and False if
            it is not required. Defaults to false.

    Returns:
        Observations found in the CSV file indexed into an IndexedObservations structure.
    """
    ret_index = KeyingObservationIndex()

    with open(path) as f:
        records_raw = csv.DictReader(f)

        for record_raw in records_raw:
            record = Observation.from_dict(record_raw)
            included = get_observation_included(require_response, record)

            if included:
                ret_index.add(
                    int(record_raw['year']),
                    str(record_raw['region']),
                    str(record_raw['subtype']),
                    record
                )

    return ret_index
