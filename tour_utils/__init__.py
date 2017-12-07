from typing import List, Dict, Any, Callable, Set, Tuple

import pandas as pd

from .framework import Activity, Trip, Tour, Zone


def duration_tie_breaker(activities: List[Activity], first_activity: Activity) -> Activity:
    max_duration = -1
    chosen_activity = None
    for act in activities:
        dur = act.duration
        if isinstance(dur, pd.Timedelta): dur = dur.seconds
        if dur > max_duration:
            chosen_activity = act
            max_duration = dur
    if chosen_activity is None:
        print(max_duration)
        print(activities)
        raise AssertionError()
    return chosen_activity


def distance_tie_breaker(activities: List[Activity], first_activity: Activity) -> Activity:
    x0, y0 = first_activity.zone.x, first_activity.zone.y

    max_distance = -1
    chosen_activity = None

    for act in activities:
        x1, y1 = act.zone.x, act.zone.y
        distance = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        if distance > max_distance:
            chosen_activity = act
            max_distance = distance
    assert chosen_activity is not None
    return chosen_activity


def duration_aggregator(activities: List[Activity], first_activity: Activity) -> Activity:
    type_ = activities[0].type_

    zonal_activities: Dict[Zone, List[Activity]] = {}
    zonal_durations: Dict[Zone, float] = {}
    for act in activities:
        if act.zone in zonal_activities:
            zonal_activities[act.zone].append(act)
            zonal_durations[act.zone] += act.duration
        else:
            zonal_activities[act.zone] = [act]
            zonal_durations[act.zone] = act.duration

    zone_with_largest_duration = None
    chosen_activities = []
    max_duration = -1
    for zone, activity_list in zonal_activities.items():
        duration = zonal_durations[zone]
        if isinstance(duration, pd.Timedelta): duration = duration.seconds
        if duration > max_duration:
            max_duration = duration
            zone_with_largest_duration = zone
            chosen_activities = activity_list

    assert zone_with_largest_duration is not None
    longest_duration = -1
    longest_activity = None
    for act in chosen_activities:
        duration = act.duration
        if isinstance(duration, pd.Timedelta): duration = duration.seconds
        if duration > longest_duration:
            longest_duration = duration
            longest_activity = act

    aggregated_activity = Activity(longest_activity.index, type_, zone_with_largest_duration)
    aggregated_activity.start_time = 0.0
    aggregated_activity.end_time = max_duration
    return aggregated_activity


def preprocess_activity_list(activities: List[Activity]):
    # Find runs of work activities, converting to "B" (business)
    # Rules that are followed:
    # 1. For runs of length 2, the second activity always becomes business.
    # 2. For runs of more than length 2, intermediate work activities become business
    runs: List[List[Activity]] = []
    current_run = []
    for a in activities:

        if a.type_ == 'W':
            current_run.append(a)
        elif len(current_run) > 0:
            runs.append(current_run)
            current_run = []
    if len(current_run) > 0: runs.append(current_run)

    for run in runs:
        if len(run) == 2:
            run[1].type_ = 'B'
        else:
            first_act = run[0]
            last_act = run[-1]
            if first_act.zone == last_act.zone:
                for a in run[1:-1]: a.type_ = 'B'
            else:
                for a in run[1:]: a.type_ = 'B'
    # Modifications are done in-place and therefore no return is needed.


class TourCreator(object):
    def __init__(self, zones: pd.DataFrame, trips: pd.DataFrame, persons: pd.DataFrame):
        self.zones_df: pd.DataFrame = zones
        self.trips_df: pd.DataFrame = trips
        self.persons_df: pd.DataFrame = persons

        self.person_ids: List[str] = ['hhid', 'pid']
        self.trip_seq: str = 'trip_id'
        self.taz_o: str = 'o_taz'
        self.taz_d: str = 'd_taz'
        self.act_o: str = 'o_activity'
        self.act_d: str = 'd_activity'
        self.weight: str = 'rer_weight'
        self.start_time: str = 'start_time'
        self.end_time: str = 'end_time'
        self.home_activity: str = 'H'
        self.x: str = 'x'
        self.y: str = 'y'
        self.zone_label = None
        self.mandatory_activities: Set[str] = {'U', 'S', 'W'}
        self.allowed_subtour_activities: Set[str] = {'W'}

        self.tie_breaker: Callable[[List[Activity], Activity], Activity] = duration_tie_breaker
        self.activity_aggregator: Callable[[List[Activity], Activity], Activity] = duration_aggregator

        self._tour_counter = 0

        self.activity_processor: Callable[List[List[Activity]], None] = preprocess_activity_list

    def run(self, presorted=False, return_df=False, skip_errors=False):
        self._validate()

        if not presorted:
            sortby = self.person_ids + [self.trip_seq]
            self.trips_df.sort_values(sortby, ascending=True, inplace=True)

        zones = self._convert_zones()
        person_triplists = self._convert_trips()
        tours, all_tours = self._convert_tours(person_triplists, zones, skip_errors)
        if not return_df: return all_tours
        return self._convert_person_tours_to_frame(all_tours)

    def _validate(self):
        assert self.taz_o in self.trips_df
        assert self.taz_d in self.trips_df
        assert self.act_o in self.trips_df
        assert self.act_d in self.trips_df
        assert self.weight in self.trips_df
        assert self.start_time in self.trips_df
        assert self.end_time in self.trips_df
        assert self.x in self.zones_df
        assert self.y in self.zones_df

        all_activitiy_types = pd.Index(self.trips_df[self.act_o].unique()) | pd.Index(self.trips_df[self.act_d].unique())
        assert self.home_activity in all_activitiy_types

    def _convert_zones(self):
        if self.zone_label is not None: assert self.zone_label in self.zones_df

        zones: Dict[int, Zone] = {}
        for record_number, (index, series) in enumerate(self.zones_df.iterrows()):
            zone_label = index if self.zone_label is None else series[self.zone_label]
            zone = Zone(zone_label, record_number, series, series[self.x], series[self.y])
            zones[zone_label] = zone
        print(f"Loaded {len(zones)} zones")
        return zones

    def _convert_trips(self):
        print("Finding closed and open trip-lists (proto-tours)")
        person_triplists: Dict[Any, List[List[Trip]]] = {}
        open_triplists: Dict[Any, List[Trip]] = {}
        for record_number, (record_id, record) in enumerate(self.trips_df.iterrows()):
            puid = tuple(record[col] for col in self.person_ids)
            trip_index = record[self.trip_seq]

            trip = Trip(trip_index, record_number, record, record[self.weight])

            if puid in open_triplists:
                open_triplists[puid].append(trip)
            else:
                open_triplists[puid] = [trip]

            if record[self.act_d] == self.home_activity:
                triplist = open_triplists.pop(puid)
                if puid not in person_triplists: person_triplists[puid] = [triplist]
                else: person_triplists[puid].append(triplist)

        for puid, triplist in open_triplists.items():
            if puid not in person_triplists:
                person_triplists[puid] = [triplist]
            else:
                person_triplists[puid].append(triplist)

        print(f"Found {len(person_triplists)} persons with trips")
        return person_triplists

    def _convert_tours(self, person_triplists: Dict[Any, List[List[Trip]]], zones, skip_errors):
        print("Converting trip lists to tours")
        person_tours = {}
        all_tours = []
        n_open = 0
        errcount = 0
        self._tour_counter = 0
        for pid, triplists in person_triplists.items():
            tour_list = []
            for triplist in triplists:
                if len(triplist) < 2:
                    n_open += 1
                    continue  # Skip tours with fewer than 2 trips

                try:
                    tour = self._make_tour_kernel(pid, triplist, zones)
                    if tour is not None:
                        tour_list.append(tour)
                        all_tours.append(tour)
                    else: n_open += 1
                except:
                    if skip_errors: errcount += 1
                    else: raise

            person_tours[pid] = tour_list
        print(f"Created {self._tour_counter} closed tours, and skipped {n_open} open tours")
        if skip_errors: print(f"Encountered {errcount} errors")
        return person_tours, all_tours

    def _make_tour_kernel(self, pid, triplist: List[Trip], zones):
        activity_list = self._assemble_activities_from_list(triplist, zones)

        if self.activity_processor is not None:
            self.activity_processor(activity_list)

        first_activity, last_activity = activity_list[0].type_, activity_list[-1].type_
        if first_activity != self.home_activity or last_activity != self.home_activity or len(triplist) < 2:
            # Skip open tours for now
            return None

        anchor_type, anchor_zone, anchor_index = self._select_anchor(activity_list)
        tour = self._create_closed_tour(pid, triplist, activity_list, anchor_type, anchor_zone, anchor_index)
        return tour

    def _assemble_activities_from_list(self, triplist: List[Trip], zones: Dict[int, Zone]) -> List[Activity]:
        activities = []

        first_trip = triplist[0]
        home_zone = zones[first_trip[self.taz_o]]
        current_activity = Activity(0, first_trip[self.act_o], home_zone)

        for i, trip in enumerate(triplist):
            current_activity.end_time = trip[self.start_time]
            activities.append(current_activity)

            zone = zones[trip[self.taz_d]]

            current_activity = Activity(i + 1, trip[self.act_d], zone)
            current_activity.start_time = trip[self.end_time]
        activities.append(current_activity)
        return activities

    def _select_anchor(self, activities: List[Activity]) -> Tuple[str, Zone, int]:
        first_activity = activities[0]

        grouper: Dict[str, List[Activity]] = {}
        for a in activities[1:-1]:  # Skip the first and last activities, which are always HOME
            key = a.type_

            if key in grouper: grouper[key].append(a)
            else: grouper[key] = [a]

        representative_activities: Dict[str, Activity] = {}
        for key, activity_list in grouper.items():
            # Create a "representative activity" for types with more than one available activity. This will be used
            # to break the tie between multiple activity types. The activity index is only used for activity types
            # that do not allow subtours

            if len(activity_list) == 1: representative_activities[key] = activity_list[0]
            else: representative_activities[key] = self.activity_aggregator(activity_list, first_activity)

        present_mandatory_activities = set(grouper.keys()) & self.mandatory_activities  # Set intersection
        if len(present_mandatory_activities) == 1:
            # Exactly one mandatory activity type
            type_ = iter(present_mandatory_activities).__next__()
            act = representative_activities[type_]
        elif len(present_mandatory_activities) > 1:
            tied_mandatory_activities = [representative_activities[type_] for type_ in present_mandatory_activities]
            act = self.tie_breaker(tied_mandatory_activities, first_activity)
        else:  # No mandatory activities present
            tied_activities = list(representative_activities.values())
            act = self.tie_breaker(tied_activities, first_activity)
        return act.type_, act.zone, act.index

    def _next_tour_id(self):
        t = self._tour_counter
        self._tour_counter += 1
        return t

    def _create_closed_tour_no_subtours(self, pid, triplist: List[Trip], activities: List[Activity], anchor_type: str,
                                        anchor_zone: Zone, anchor_index: int) -> Tour:
        tour = Tour(pid, self._next_tour_id())
        first_act = activities[0]
        tour.p_act, tour.p_zone, tour.start_time = first_act.type_, first_act.zone, first_act.end_time
        tour.a_zone, tour.a_act = anchor_zone, anchor_type

        current_stop_list, current_triplist = [], []
        direction_is_outbound = True
        for i, trip in enumerate(triplist[:-1]):  # Exclude the last trip and activity
            activity = activities[i + 1]  # The activity at the end of this trip
            current_triplist.append(trip)  # Never lose trips

            if activity.index == anchor_index:
                assert direction_is_outbound

                direction_is_outbound = False
                tour.outbound_acts = current_stop_list
                tour.outbound_trips = current_triplist
            else:
                # Only add activities to the list if they are intermediate stops
                current_stop_list.append(activity)

    @staticmethod
    def _tour_status_with_subtours(activity: Activity, anchor_type: str, anchor_zone: Zone, anchor_index: int,
                                   direction_is_outbound) -> str:
        if activity.type_ == anchor_type and activity.zone == anchor_zone:
            if not direction_is_outbound: return "SUBTOUR_RETURN"
            return "TURN_AROUND"
        return "CONTINUE"

    @staticmethod
    def _tour_status_no_subtours(activity: Activity, anchor_type: str, anchor_zone: Zone, anchor_index: int,
                                direction_is_outbound) -> str:
        if activity.index == anchor_index:
            assert direction_is_outbound
            return "TURN_AROUND"
        return "CONTINUE"

    def _create_closed_tour(self, pid, triplist: List[Trip], activities: List[Activity], anchor_type: str,
                            anchor_zone: Zone, anchor_index: int) -> Tour:
        primary_tour = Tour(pid, self._next_tour_id())
        first_act = activities[0]
        primary_tour.p_act, primary_tour.p_zone, primary_tour.start_time = first_act.type_, first_act.zone, first_act.end_time
        primary_tour.a_zone, primary_tour.a_act = anchor_zone, anchor_type

        status_check = self._tour_status_with_subtours if anchor_type in self.allowed_subtour_activities else self._tour_status_no_subtours

        current_stop_list, current_triplist = [], []
        direction_is_outbound = True
        for i, trip in enumerate(triplist[:-1]):  # Exclude the last trip and activity
            activity = activities[i + 1]  # The activity at the end of this trip

            current_triplist.append(trip)  # Never lose trips

            status = status_check(activity, anchor_type, anchor_zone, anchor_index, direction_is_outbound)
            if status == "SUBTOUR_RETURN":
                subtour = Tour(pid, self._next_tour_id())
                subtour.p_act, subtour.p_zone = anchor_type, anchor_zone
                subtour.parent_tour = primary_tour

                subtour_anchor = self.tie_breaker(current_stop_list, current_stop_list[0])
                subtour.a_act, subtour.a_zone = subtour_anchor.type_, subtour_anchor.zone
                subtour.start_time = current_stop_list[0].start_time
                subtour.end_time = current_stop_list[-1].end_time

                anchor_index = subtour_anchor.index

                subtour.outbound_acts = current_stop_list[:anchor_index]
                subtour.inbound_acts = current_stop_list[anchor_index + 1:]
                subtour.outbound_trips = current_triplist[:anchor_index]
                subtour.inbound_trips = current_triplist[anchor_index:]

                primary_tour.subtours.append(subtour)

                # Create new accumulators
                current_stop_list = []
                current_triplist = []
            elif status == "TURN_AROUND":
                direction_is_outbound = False
                primary_tour.outbound_acts = current_stop_list
                primary_tour.outbound_trips = current_triplist

                # Create new accumulators
                current_stop_list = []
                current_triplist = []
            elif status == "CONTINUE":
                # Only add activities to the list if they are intermediate stops
                current_stop_list.append(activity)
            else:
                raise RuntimeError(status)

        # Cleanup the remaining accumulators
        last_trip = triplist[-1]
        last_act = activities[-1]

        current_triplist.append(last_trip)
        primary_tour.inbound_trips = current_triplist
        primary_tour.inbound_acts = current_stop_list
        primary_tour.end_time = last_act.start_time

        return primary_tour

    def _convert_person_tours_to_frame(self, tours: List[Tour]) -> pd.DataFrame:
        simple_columns = [
            "p_act", "a_act", "start_time", "end_time", "weight", "n_stops_out", "n_stops_in",
            "activity_pattern"
        ]
        all_columns = simple_columns + ["n_subtours", "parent_tour", "p_zone", "a_zone"] + self.person_ids

        index = []
        proto_frame = {key: [] for key in all_columns}

        for tour in tours:
            pid_list = list(tour.pid)
            for col in simple_columns: proto_frame[col].append(getattr(tour, col))

            index.append(tour.tour_id)
            proto_frame['parent_tour'].append(-1)
            proto_frame['n_subtours'].append(len(tour.subtours))
            proto_frame['p_zone'].append(tour.p_zone.number if tour.p_zone is not None else -1)
            proto_frame['a_zone'].append(tour.a_zone.number if tour.a_zone is not None else -1)
            for col_name, val in zip(self.person_ids, pid_list): proto_frame[col_name].append(val)

            for subtour in tour.subtours:
                for col in simple_columns: proto_frame[col].append(getattr(subtour, col))
                index.append(subtour.tour_id)
                proto_frame['parent_tour'].append(tour.tour_id)
                proto_frame['n_subtours'].append(0)
                proto_frame['p_zone'].append(subtour.p_zone.number if subtour.p_zone is not None else -1)
                proto_frame['a_zone'].append(subtour.a_zone.number if subtour.a_zone is not None else -1)
                for col_name, val in zip(self.person_ids, pid_list): proto_frame[col_name].append(val)

        return pd.DataFrame(proto_frame, index=index)
