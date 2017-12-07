from typing import List

import pandas as pd


class Trip(object):
    def __init__(self, index, table_pointer, record: pd.Series, weight: float):
        self.index, self.table_pointer, self._record, self.weight = index, table_pointer, record, weight

    def __getattr__(self, item):
        return getattr(self._record, item)

    def __getitem__(self, item): return self._record[item]


class Zone(object):
    def __init__(self, number, table_pointer, record: pd.Series, x: float, y: float):
        self.number, self.table_pointer, self._record = number, table_pointer, record
        self.x, self.y = x, y

    def __getattr__(self, item): return getattr(self._record, item)

    def __getitem__(self, item): return self._record[item]

    def __repr__(self):
        return f"Zone({self.number} @ [{self.x}, {self.y}]"

    def __str__(self): return str(self.number)

    def __eq__(self, other):
        return self.number == other.number

    def __hash__(self):
        return hash(self.number)


class Activity(object):
    def __init__(self, index: int, type_: str, zone: Zone=None):
        self.index: int = index
        self.type_: str = type_
        self.zone: Zone = zone

        self.start_time: float = float('-inf')
        self.end_time: float = float('inf')

    @property
    def duration(self): return self.end_time - self.start_time

    def __repr__(self):
        return f"Activity({self.index} {self.type_} [@ {self.zone}] from {self.start_time} to {self.end_time}"


class Tour(object):
    def __init__(self, pid, tour_id):
        self.pid = pid
        self.tour_id = tour_id
        self.closed = False

        self.p_act: str = None
        self.a_act: str = None
        self.p_zone: Zone = None
        self.a_zone: Zone = None
        self.start_time = float('-inf')
        self.end_time = float('inf')

        self.outbound_acts: List[Activity] = []
        self.inbound_acts: List[Activity] = []
        self.outbound_trips: List[Trip] = []
        self.inbound_trips: List[Trip] = []

        self.parent_tour: 'Tour' = None
        self.subtours: List['Tour'] = []

        self._cached_weight = None

    @property
    def n_stops_out(self) -> int: return len(self.outbound_acts)

    @property
    def n_stops_in(self) -> int: return len(self.inbound_acts)

    @property
    def n_activities(self) -> int:
        sub_lengths = sum(t.n_activities - 2 for t in self.subtours)
        return 2 + self.n_stops_out + self.n_stops_in + sub_lengths

    @property
    def n_trips(self) -> int:
        sub_lengths = sum(t.n_trips for t in self.subtours)
        return len(self.outbound_trips) + len(self.inbound_trips) + sub_lengths

    @property
    def weight(self) -> float:
        if self._cached_weight is not None: return self._cached_weight

        my_weight = 0
        n_trips = 0
        for trip in self.outbound_trips + self.inbound_trips:
            n_trips += 1
            my_weight += trip.weight
        my_weight *= n_trips

        sub_weight = 0.0
        for tour in self.subtours:
            n_subtrips = tour.n_trips
            sub_weight += tour.weight * n_subtrips
            n_trips += n_subtrips

        weight = (my_weight + sub_weight) / n_trips
        self._cached_weight = weight
        return weight

    @property
    def activity_pattern(self) -> str:
        components = [self.p_act]
        for act in self.outbound_acts: components.append(act.type_)
        for tour in self.subtours:
            components.append(self.a_act)
            for act in tour.outbound_acts: components.append(act.type_)
            for act in tour.inbound_acts: components.append(act.type_)
        components.append(self.a_act)
        for act in self.inbound_acts: components.append(act.type_)
        components.append(self.p_act)
        return ''.join(components)
