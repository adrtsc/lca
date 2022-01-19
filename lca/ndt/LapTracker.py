# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:15:09 2020
@author: Adrian
"""

import pickle
import logging
import numpy.matlib

import numpy as np
import pandas as pd
import networkx as nx

from progress.bar import Bar
from skimage.measure import regionprops_table
from sklearn.metrics.pairwise import paired_distances
from scipy.spatial import cKDTree
from sslap import auction_solve
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from numpy.random import uniform


def load_tracker(path_to_tracker):
    '''
    Loads a tracker objects from the specified path (includes filename)
    Parameters
    ----------
    path_to_tracker : string
        path where the tracker object is saved.
    Returns
    -------
    tracker : object
        tracker object
    '''

    with open(path_to_tracker, "rb") as fp:
        tracker = pickle.load(fp)

    return tracker


class LapTracker():
    """
    A class used to represent an a linear assignment problem tracker
    The tracking algorithm used is based on
    ...
    Attributes
    ----------
    df : pd.DataFrame
        a pandas dataframe containing the x, y, and t coordinates of objects
    max_distance : int
        maximal distance to allow particle linking between frames
    time_window : int
        maximal memory of the tracker to link track segments
    identifiers : list
        list of column names for coordinates in df and label column
        (e.g. ['x_coord', 'y_coord', 'timepoint', 'labels'])
    max_split_distance : int
        maximal distance to link segment starts to segment middlepoints
    allow_merging : bool
        indicates whether object merging is allowed or not (default: False)
    allow_splitting: bool
        indicates whether object splitting is allowed or not (default: True)
    segments : list
        contains the unique ids of the members of the detected segments
    segments_by_label : list
        contains the labels of the members of the detected segments
    tracks : list
        contains the unique ids of the members of the detected tracks
    tracks_by_label : list
        contains the labels of the members of the detected tracks
    Methods
    -------
    track_df(df, identifiers)
        tracks the objects in df and assigns 3 additional collumns
        (unique_id, segment_id and track_id).
    track_label_images(label_stack, intensity_stack(optional))
        tracks the objects in the movie and computes centroid/track features.
    """

    def __init__(self,
                 max_distance,
                 time_window,
                 max_split_distance,
                 max_gap_closing_distance,
                 allow_merging=False,
                 allow_splitting=True,
                 cost_factor=0.9):
        """
        Parameters
        ----------
        max_distance : int
            maximal distance to allow particle linking between frames
        time_window : int
            maximal memory of the tracker to link track segments
        max_split_distance : int
            maximal distance to link segment starts to segment middlepoints
        max_gap_closing_distance: int
            maximal distance to close gaps
        allow_merging : bool
            indicates whether object merging is allowed or not
            (default: False)
        allow_splitting: bool
            indicates whether object splitting is allowed or not
            (default: True)
        """

        self.max_distance = max_distance
        self.max_split_distance = max_split_distance
        self.max_gap_closing_distance = max_gap_closing_distance
        self.time_window = time_window
        self.allow_merging = allow_merging
        self.allow_splitting = allow_splitting
        self.link_costs = []
        self.global_costs = []
        self.cost_factor = cost_factor

    def __get_frame_linking_matrix(self,
                                   t_coords,
                                   t1_coords):
        """Generates cost matrix for object linking"""

        number_of_objects_t_0 = len(t_coords)
        number_of_objects_t_1 = len(t1_coords)
        n_objects = number_of_objects_t_0 + number_of_objects_t_1
        frame_linking_matrix = lil_matrix((n_objects, n_objects))

        # calculate sparse distance matrix
        # (all elements above max_distance are set to zero)
        kd_tree_t = cKDTree(t_coords)
        kd_tree_t1 = cKDTree(t1_coords)

        dist = kd_tree_t.sparse_distance_matrix(kd_tree_t1, self.max_distance)

        frame_linking_matrix[0:number_of_objects_t_0,
        0:number_of_objects_t_1] = dist

        # calculate the lower right block
        lower_right_block = np.transpose(dist) * 0.0001
        frame_linking_matrix[
        number_of_objects_t_0:n_objects,
        number_of_objects_t_1:n_objects] = lower_right_block

        # calculate non-link matrix
        if self.link_costs == []:
            non_link_cost = self.max_distance
        else:
            non_link_cost = 1.05 * np.max(self.link_costs)

        non_link = lil_matrix((number_of_objects_t_0,
                               number_of_objects_t_0))

        non_link.setdiag(non_link_cost)

        frame_linking_matrix[0:number_of_objects_t_0,
        number_of_objects_t_1:n_objects] = non_link

        non_link_1 = lil_matrix((number_of_objects_t_1,
                                 number_of_objects_t_1))

        non_link_1.setdiag(non_link_cost)

        frame_linking_matrix[number_of_objects_t_0:n_objects,
        0:number_of_objects_t_1] = non_link_1

        return frame_linking_matrix

    def __get_intensity_matrix(self, split_dist):
        '''Generates intensity matrix for object splitting/merging'''
        intensity_matrix = lil_matrix(np.shape(split_dist))
        r, c = split_dist.nonzero()
        bar = Bar('calculating intensity matrix for plausible split events',
                  max=len(r),
                  check_tty=False, hide_cursor=False)
        for split in range(0, len(r)):
            # get intensity of current middlepoint
            mp_int = self.segment_middlepoints['sum_intensity'].iloc[r[split]]
            mp_timepoint = self.segment_middlepoints['timepoint'].iloc[
                r[split]]
            mp_segment_id = self.segment_middlepoints['segment_id'].iloc[
                r[split]]
            next_mp_int = np.array(
                self.segment_middlepoints['sum_intensity'].loc[
                    (self.segment_middlepoints['timepoint'] ==
                     mp_timepoint) &
                    (self.segment_middlepoints['segment_id'] ==
                     mp_segment_id)])
            next_mp_int = next_mp_int[0]
            # get intensity of current segment start
            start_int = self.start_points['sum_intensity'].iloc[c[split]]
            intensity_ratio = (mp_int / (next_mp_int + start_int))
            if intensity_ratio > 1:
                intensity_matrix[r[split], c[split]] = intensity_ratio
            elif intensity_ratio < 1:
                intensity_matrix[r[split], c[split]] = (
                        intensity_ratio ** -self.intensity_weight)
            bar.next()
        bar.finish()
        return intensity_matrix

    def __get_segment_linking_matrix(self):
        """Generates cost matrix for segment linking"""

        segment_linking_matrix = lil_matrix((
            (self.number_of_segments +
             self.number_of_segment_middlepoints) * 2,
            (self.number_of_segments +
             self.number_of_segment_middlepoints) * 2))

        # gap_closing_matrix

        # find a matrix to define time window

        kd_tree_tend = cKDTree(self.end_points[[self.identifiers[2],
                                                'filler']])
        kd_tree_tstart = cKDTree(self.end_points[[self.identifiers[2],
                                                  'filler']])

        temp_dist = kd_tree_tend.sparse_distance_matrix(kd_tree_tstart,
                                                        self.time_window)

        temp_dist[temp_dist.nonzero()] = 1
        # get the distance matrix of start and end points and delete all
        # values that are outside of gap closing distance

        kd_tree_end = cKDTree(self.end_points[[self.identifiers[0],
                                               self.identifiers[1]]])

        kd_tree_start = cKDTree(self.start_points[[self.identifiers[0],
                                                   self.identifiers[1]]])

        dist = kd_tree_end.sparse_distance_matrix(
            kd_tree_start,
            self.max_gap_closing_distance)

        # set all values that are outside of the time window to 0
        # done here by multiplying the binarized array temp_dist with dist

        dist = dist.multiply(temp_dist).tolil()

        # set diagonal to 0

        dist.setdiag(0)

        segment_linking_matrix[0:self.number_of_segments,
        0:self.number_of_segments] = dist

        # add costs for gap closing to global costs

        self.global_costs.extend(list(
            segment_linking_matrix[segment_linking_matrix > 0].data[0]))

        # merge matrix

        # merging is at the moment not implemented, so no merging allowed

        # center_matrix: all disallowed

        # split_matrix

        kd_tree_middle = cKDTree(
            self.segment_middlepoints[[self.identifiers[0],
                                       self.identifiers[1]]])

        kd_tree_start = cKDTree(self.start_points[[self.identifiers[0],
                                                   self.identifiers[1]]])

        split_dist = kd_tree_middle.sparse_distance_matrix(
            kd_tree_start,
            self.max_split_distance)

        # find splits that are close enough at t-1

        matrix_a = np.transpose(
            numpy.matlib.repmat(self.segment_middlepoints[self.identifiers[2]],
                                self.number_of_segments, 1))
        matrix_b = numpy.matlib.repmat(self.start_points[self.identifiers[2]],
                                       self.number_of_segment_middlepoints, 1)

        matrix_c = matrix_b - matrix_a

        split_dist = split_dist.tolil()

        split_dist[(matrix_c > 1) | (matrix_c <= 0)] = 0

        if split_dist.nonzero()[0].size == 0:
            logging.warnings.warn('no plausible splits detected, consider increasing max_split_distance')

        # include intensity comparison if requested
        if "sum_intensity" in self.df:
            intensity_matrix = self.__get_intensity_matrix(split_dist)
            split_dist = split_dist.multiply(intensity_matrix).tolil()
        else:
            split_dist = split_dist.tolil()

        segment_linking_matrix[
        self.number_of_segments:(self.number_of_segments +
                                 self.number_of_segment_middlepoints),
        0:self.number_of_segments] = split_dist

        # add costs for splitting to global costs

        self.global_costs.extend(list(split_dist[split_dist > 0].data[0]))

        # termination matrix

        termination_matrix = lil_matrix(
            (self.number_of_segments,
             (self.number_of_segments +
              self.number_of_segment_middlepoints)))

        self.global_costs = np.array(self.global_costs)
        if self.global_costs.size > 0:
            termination_cost = np.quantile(self.global_costs, self.cost_factor)
        else:
            termination_cost = 1

        termination_matrix.setdiag(termination_cost)

        segment_linking_matrix[
        0:self.number_of_segments,
        (self.number_of_segments +
         self.number_of_segment_middlepoints):] = termination_matrix

        # initiation matrix

        initiation_matrix = lil_matrix(((self.number_of_segments +
                                         self.number_of_segment_middlepoints),
                                        self.number_of_segments))

        initiation_matrix.setdiag(termination_cost)

        segment_linking_matrix[(self.number_of_segments +
                                self.number_of_segment_middlepoints):,
        0:self.number_of_segments] = initiation_matrix

        # merge_refusal_matrix

        merge_refusal_matrix = lil_matrix(
            ((self.number_of_segments +
              self.number_of_segment_middlepoints),
             self.number_of_segment_middlepoints))

        allowed = lil_matrix((self.number_of_segment_middlepoints,
                              self.number_of_segment_middlepoints))

        diagonal = self.average_displacement[np.array(
            [self.segment_middlepoints['segment_id']])] ** 2

        allowed.setdiag(diagonal.tolist()[0])

        merge_refusal_matrix[self.number_of_segments:,
        0:] = allowed

        segment_linking_matrix[(
                                       self.number_of_segments + self.number_of_segment_middlepoints):,
        self.number_of_segments:(
                self.number_of_segments +
                self.number_of_segment_middlepoints)] = merge_refusal_matrix

        split_refusal_matrix = lil_matrix(
            (self.number_of_segment_middlepoints,
             (self.number_of_segments +
              self.number_of_segment_middlepoints)))

        split_refusal_matrix[0:,
        self.number_of_segments:] = allowed

        segment_linking_matrix[
        self.number_of_segments:(self.number_of_segment_middlepoints +
                                 self.number_of_segments),
        (self.number_of_segments +
         self.number_of_segment_middlepoints):] = split_refusal_matrix

        # get lower right block

        lower_right_block = segment_linking_matrix[
                            0:(self.number_of_segments +
                               self.number_of_segment_middlepoints),
                            0:(self.number_of_segments +
                               self.number_of_segment_middlepoints)]

        lower_right_block = np.transpose(lower_right_block) * 0.0001

        segment_linking_matrix[
        (self.number_of_segments +
         self.number_of_segment_middlepoints):,
        (self.number_of_segments +
         self.number_of_segment_middlepoints):] = lower_right_block

        return segment_linking_matrix

    def __get_track_segments(self):
        """Computes track segments from segment linking matrix"""
        bar = Bar('linking objects across time', max=self.number_of_timepoints,
                  check_tty=False, hide_cursor=False)
        for timepoint in range(0, self.number_of_timepoints):

            features_t0 = self.df.loc[
                self.df[self.identifiers[2]] == timepoint].sort_values(
                'unique_id')
            features_t1 = self.df.loc[
                self.df[self.identifiers[2]] == timepoint + 1].sort_values(
                'unique_id')

            number_of_objects_t_0 = len(features_t0)
            number_of_objects_t_1 = len(features_t1)

            if (number_of_objects_t_0 != 0) & (number_of_objects_t_1 != 0):

                t_coords = features_t0[[self.identifiers[0],
                                        self.identifiers[1]]]

                t1_coords = features_t1[[self.identifiers[0],
                                         self.identifiers[1]]]
                # get cost matrix

                self.cost_matrix_linking = self.__get_frame_linking_matrix(
                    t_coords,
                    t1_coords)

                self.input_solver_1 = self.cost_matrix_linking.copy().tocsr()

                self.input_solver_1[self.input_solver_1.nonzero()] = self.input_solver_1.max() / self.input_solver_1[
                    self.input_solver_1.nonzero()]

                # get optimal linking from cost matrix
                sol = auction_solve(coo_mat=self.input_solver_1.tocoo(),
                                    problem='max', cardinality_check=False)

                col_ind = list(sol.values())[0]
                row_ind = np.arange(0, len(col_ind))

                matches = np.where(col_ind < number_of_objects_t_1)[0]

                for match in matches:
                    if ((col_ind[match] < number_of_objects_t_1) &
                            (row_ind[match] < number_of_objects_t_0)):
                        self.adjacency_matrix[
                            features_t0['unique_id'].iloc[row_ind[match]],
                            features_t1['unique_id'].iloc[col_ind[match]]] = 1

                link_matrix = self.cost_matrix_linking[
                              0: number_of_objects_t_0,
                              0: number_of_objects_t_1]

                self.object_row_index = row_ind
                self.object_col_index = col_ind

                # add costs of the links made at this timepoint
                self.link_costs = self.link_costs + list(
                    link_matrix[link_matrix.nonzero()].data[0])
            bar.next()
        bar.finish()
        # compute a weakly connected directed graph from the
        # adjacency matrix. I used the graph approach because
        # it's relatively simple to get the single tracks out.

        self.G = nx.DiGraph(self.adjacency_matrix)
        self.number_of_segments = nx.number_weakly_connected_components(self.G)
        self.segments = [sorted(c) for c in sorted(
            nx.weakly_connected_components(self.G),
            key=len,
            reverse=True)]

        self.segments_by_label = [list(self.df[self.identifiers[3]].iloc[
                                           sorted(c)]) for c in sorted(nx.weakly_connected_components(self.G),
                                                                       key=len,
                                                                       reverse=True)]

        # add column for segment ids, identifiers for start/end/middlepoints

        is_start = np.zeros([self.number_of_objects])
        is_end = np.zeros([self.number_of_objects])
        is_middlepoint = np.zeros([self.number_of_objects])

        starts = [item[0] for item in self.segments]

        ends = [item[-1] for item in self.segments]

        is_start[starts] = 1
        is_end[ends] = 1
        is_middlepoint[(is_start == 0) &
                       (is_end == 0)] = 1
        # get segment index
        length = max(map(len, self.segments))
        a = np.array([xi + [np.nan] * (length - len(xi)) for xi in self.segments])
        a_h = np.size(a, 1)
        a = np.reshape(a, a.size)
        b = np.array(list(range(0, self.number_of_objects)))

        sorter = np.argsort(a)
        ndx = sorter[np.searchsorted(a, b, sorter=sorter)]
        segment_ids = np.floor(ndx / a_h).astype(int)

        self.df['segment_id'] = segment_ids
        self.df['is_start'] = is_start
        self.df['is_end'] = is_end
        self.df['is_middlepoint'] = is_middlepoint
        self.df['filler'] = np.zeros([self.number_of_objects])

    def __close_gaps(self):
        """Deals with gaps/merging/splitting based on to gap closing matrix"""

        # get start and end points of the detected segments

        self.start_points = self.df.loc[self.df['is_start'] == 1].sort_values(
            'segment_id')
        self.end_points = self.df.loc[self.df['is_end'] == 1].sort_values(
            'segment_id')

        # get the segment middle points

        segment_middlepoints = self.df.loc[self.df['is_middlepoint'] == 1]
        self.segment_middlepoints = segment_middlepoints.sort_values(
            'unique_id')
        self.number_of_segment_middlepoints = len(segment_middlepoints)

        # calculate average displacement for each segment
        # this is later used to compute the cost matrix

        print('calculating average displacement per segment')

        def get_average_displacement(df):
            if len(df) > 1:
                test = paired_distances(
                    df[[self.identifiers[0],
                        self.identifiers[1]]].iloc[1:, :],
                    df[[self.identifiers[0],
                        self.identifiers[1]]].shift().iloc[1:, :])
                return np.mean(test)
            else:
                pass

        self.average_displacement = np.array(
            self.df.groupby('segment_id').apply(get_average_displacement))

        self.average_displacement[
            np.isnan(self.average_displacement)] = np.nanmean(
            self.average_displacement)

        self.cost_matrix_gap_closing = self.__get_segment_linking_matrix()
        self.input_solver_2 = self.cost_matrix_gap_closing.copy().tocsr()
        self.input_solver_2[self.input_solver_2.nonzero()] = self.input_solver_2.max() / self.input_solver_2[
            self.input_solver_2.nonzero()]
        # self.cost_matrix_gap_closing[self.cost_matrix_gap_closing.nonzero()] = np.abs(
        #             self.cost_matrix_gap_closing[self.cost_matrix_gap_closing.nonzero()] - (self.cost_matrix_gap_closing.max()-1))

        Cs = coo_matrix(self.input_solver_2.tocoo())
        sol = auction_solve(coo_mat=Cs, problem='max', cardinality_check=False)

        col_ind = list(sol.values())[0]
        row_ind = np.arange(0, len(col_ind))

        # get unique ids of the middlepoints that undergo a split

        sources = np.where(col_ind < self.number_of_segments)[0]
        sources = sources[(sources > self.number_of_segments) &
                          (sources < (self.number_of_segments +
                                      self.number_of_segment_middlepoints))]
        target_segment_starts = col_ind[np.array(sources)]
        sources = sources - self.number_of_segments
        sources_unique_id = np.array(
            self.segment_middlepoints['unique_id'].iloc[sources], dtype='int')

        # get unique id of the segment starting points

        target_unique_id = np.array(
            self.start_points['unique_id'].iloc[
                np.array(target_segment_starts)], dtype='int')

        self.adjacency_matrix[sources_unique_id, target_unique_id] = 1

        # gap closing

        sources = np.where(col_ind < self.number_of_segments)[0]
        sources = sources[(sources >= 0) &
                          (sources < self.number_of_segments)]
        sources_unique_id = np.array(
            self.end_points['unique_id'].iloc[np.array(sources)], dtype='int')

        # get unique id of the segment starting points
        target_segment_starts = col_ind[np.array(sources)]
        target_unique_id = np.array(
            self.start_points['unique_id'].iloc[
                np.array(target_segment_starts)], dtype='int')

        self.adjacency_matrix[sources_unique_id, target_unique_id] = 1
        self.segment_row_index = row_ind
        self.segment_col_index = col_ind

    def track_df(self, df, identifiers, modulate_centroids=True):
        """
        Tracks the objects in df
        Assigns 3 additional columns to df
        (unique_id, segment_id and track_id).
        Parameters
        ----------
        df : pd.DataFrame
            a pandas dataframe containing the x, y, and t coordinates
            of objects
        identifiers : list
            list of column names for coordinates in df and label column
            (e.g. ['x_coord', 'y_coord', 'timepoint', 'labels'])
        modulate_centroids: Boolean
            whether centroids should be slightly shifted for each timepoint. This is needed in case the segmentations
            stay constant across timepoints - otherwise the distance between their centroids will be NaN, which messes
            up the tracking.
        """
        self.df = df.copy()
        self.identifiers = identifiers
        self.number_of_objects = len(df)
        self.number_of_timepoints = np.max(np.unique(df[identifiers[2]]))
        self.adjacency_matrix = lil_matrix((self.number_of_objects,
                                            self.number_of_objects))

        if modulate_centroids:
            # modulate centroids
            self.df['centroid-0'] = self.df['centroid-0'] + uniform(size=len(self.df))
            self.df['centroid-1'] = self.df['centroid-1'] + uniform(size=len(self.df))
        # add unique identifiers to df
        self.df['unique_id'] = list(range(0, self.number_of_objects))
        # link timepoints to get segments
        self.__get_track_segments()
        # try to link the segments among themselves
        print('linking track segments across timepoints')
        self.__close_gaps()
        # get the final tracks
        self.G2 = nx.DiGraph(self.adjacency_matrix)
        self.number_of_tracks = nx.number_weakly_connected_components(self.G2)
        self.tracks = [sorted(c) for c in sorted(
            nx.weakly_connected_components(self.G2),
            key=len,
            reverse=True)]

        self.tracks_by_label = [list(self.df[self.identifiers[3]].iloc[
                                         sorted(c)]) for c in sorted(
            nx.weakly_connected_components(self.G2),
            key=len,
            reverse=True)]

        # add column for track ids
        length = max(map(len, self.tracks))
        a = np.array([xi + [np.nan] * (length - len(xi)) for xi in self.tracks])
        a_h = np.size(a, 1)
        a = np.reshape(a, a.size)
        b = np.array(list(range(0, self.number_of_objects)))

        sorter = np.argsort(a)
        ndx = sorter[np.searchsorted(a, b, sorter=sorter)]
        track_ids = np.floor(ndx / a_h).astype(int)

        self.df['track_id'] = track_ids

        # get splits
        graph = self.G2.copy()
        splitting_nodes = [node for node in graph if graph.degree(node) >= 3]

        graph.remove_nodes_from(splitting_nodes)

        self.paths = [sorted(c) for c in sorted(
            nx.weakly_connected_components(graph),
            key=len,
            reverse=True)]

        # add column for path ids
        length = max(map(len, self.paths))
        a = np.array([xi + [np.nan] * (length - len(xi)) for xi in self.paths])
        a_h = np.size(a, 1)
        a = np.reshape(a, a.size)
        b = np.array(list(range(0, self.number_of_objects)))

        sorter = np.argsort(a)
        ndx = sorter[np.searchsorted(a, b, sorter=sorter)]
        path_ids = np.floor(ndx / a_h).astype(int)

        self.df['path_id'] = path_ids

        return (track_ids)

    def __switch_labels(self, label_stack):
        relabeled_movie = np.zeros(np.shape(label_stack), dtype='uint16')
        bar = Bar('switching labels', max=self.number_of_timepoints,
                  check_tty=False, hide_cursor=False)
        for t in range(0, self.number_of_timepoints):
            label_image = label_stack[t, :, :]
            old_labels = self.df['label'].loc[
                self.df.timepoint == t]
            new_labels = self.df['track_id'].loc[
                             self.df.timepoint == t] + 1
            arr = np.zeros(label_image.max() + 1, dtype='uint16')
            arr[old_labels] = new_labels
            relabeled_movie[t, :, :] = arr[label_image]
            bar.next()
        bar.finish()

        return relabeled_movie

    def localize_objects(self, label_stack, intensity_stack=None):
        ''' localizes objects in label images
        Returns:
        -------
            df: pd.DataFrame
                DataFrame containing x, y, t coordinates and labels of objects
                (optionally also contains intensity measurements)
        '''

        df = pd.DataFrame()
        # measure centroids of objects at all timepoints
        bar = Bar('measuring centroids', max=self.number_of_timepoints,
                  check_tty=False, hide_cursor=False)

        if intensity_stack is None:
            for t in range(0, self.number_of_timepoints):
                current_features = regionprops_table(
                    label_stack[t, :, :],
                    properties=['label',
                                'centroid'])
                current_features['timepoint'] = t
                current_features = pd.DataFrame(current_features)
                df = df.append(current_features)
                bar.next()
            bar.finish()
        else:
            for t in range(0, self.number_of_timepoints):
                current_features = regionprops_table(
                    label_stack[t, :, :],
                    intensity_stack[t, :, :],
                    properties=['label',
                                'centroid',
                                'mean_intensity',
                                'area'])
                current_features['timepoint'] = t
                current_features = pd.DataFrame(current_features)
                current_features['sum_intensity'] = (
                        current_features['area'] *
                        current_features['mean_intensity'])
                df = df.append(current_features)
                bar.next()
            bar.finish()
        return df

    def draw_track_graph(self, track_id):
        subgraph = self.G2.subgraph(nodes=self.tracks[track_id])
        nx.draw_kamada_kawai(subgraph, with_labels=True)

    def track_label_images(self, label_stack, intensity_stack=None,
                           intensity_weight=2, modulate_centroids=True):
        '''
        Tracks objects in label images over timepoints
        Takes a 3D numpy array (t, x, y) with labelled objects and tries
        to link them between timepoints.
        Parameters
        ----------
        label_stack: np.array
            3D numpy array of labelled objects (t, x, y)
        intensity_stack: np.array
            3D numpy array of intensity images (t, x, y) (default: None)
        Returns
        -------
        tracked_movie: np.array
            3D numpy array with objects relabelled as their track id
        df: pd.DataFrame
            feature measurements used for tracking
        '''

        self.intensity_weight = intensity_weight
        self.number_of_timepoints = np.size(label_stack, 0)
        df = self.localize_objects(label_stack, intensity_stack)
        # track the objects in the df
        df['track_id'] = self.track_df(df, ['centroid-0', 'centroid-1', 'timepoint', 'label'],
                                       modulate_centroids=modulate_centroids)
        # relabel the movie according to the track id
        self.relabeled_movie = self.__switch_labels(label_stack)

        return self.relabeled_movie, df

    def save(self, path_to_tracker):
        '''
        Saves the tracker object at the specified path
        Parameters
        ----------
        path_to_tracker: string
            path where tracker should be saved (including filename)
        Returns
        -------
        None.
        '''
        self.path_to_tracker = path_to_tracker

        with open(self.path_to_tracker, "wb") as fp:  # Pickling
            pickle.dump(self, fp)

        print('saved tracker object at %s' % path_to_tracker)