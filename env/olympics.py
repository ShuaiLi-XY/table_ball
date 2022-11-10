import random
import math
import os
import sys
from pathlib import Path

# CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
# olympics_path = os.path.join(CURRENT_PATH)
# sys.path.append(olympics_path)

from olympics_engine.generator import create_scenario
from olympics_engine.scenario import billiard_joint, table_hockey, football, wrestling, curling_competition, \
    Running_competition

from olympics_engine.AI_olympics import AI_Olympics

from utils.box import Box
from .simulators.game import Game

import numpy as np


class Olympics(Game):
    def __init__(self, conf, seed=None):
        super(Olympics, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = seed
        self.set_seed()

        if self.game_name == 'billiard':
            Gamemap = create_scenario("billiard-joint")
            self.env_core = billiard_joint(Gamemap)
        elif self.game_name == 'tablehockey':
            Gamemap = create_scenario("table-hockey")
            self.env_core = table_hockey(Gamemap)
        elif self.game_name == 'football':
            Gamemap = create_scenario("football")
            self.env_core = football(Gamemap)
        elif self.game_name == 'wrestling':
            Gamemap = create_scenario("wrestling")
            self.env_core = wrestling(Gamemap)
        elif self.game_name == 'curling':
            Gamemap = create_scenario("curling-competition")
            self.env_core = curling_competition(Gamemap)
        elif self.game_name == 'integrated':
            self.env_core = AI_Olympics(random_selection=True, minimap=False)
        elif self.game_name == 'running':
            self.num_map = conf['map_num']
            map_index_seq = list(range(1, conf['map_num'] + 1))
            weight = np.array(map_index_seq)
            weights = weight / weight.sum()
            rand_map_idx = random.choices(map_index_seq, weights)[0]
            Gamemap = create_scenario("running-competition")
            self.env_core = Running_competition(meta_map=Gamemap, map_id=rand_map_idx)

            self.env_core.VIEW_BACK = -self.env_core.map['agents'][0].r / self.env_core.map['agents'][0].visibility
            self.env_core.obs_boundary_init = list()
            self.env_core.obs_boundary = self.env_core.obs_boundary_init

            for index, item in enumerate(self.env_core.map["agents"]):
                position = item.position_init
                r = item.r
                if item.type == 'agent':
                    visibility = item.visibility
                    boundary = self.env_core.get_obs_boundaray(position, r, visibility)
                    self.env_core.obs_boundary_init.append(boundary)
                else:
                    self.env_core.obs_boundary_init.append(None)

        self.max_step = int(conf['max_step'])
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space

        self.step_cnt = 0
        self.won = {}
        self.n_return = [0] * self.n_player
        self.init_info = {}

        if self.game_name != 'integrated':
            map_element = []
            for i in self.env_core.map['objects']:
                if i.type == 'arc':
                    map_element.append(
                        [i.type, i.init_pos, i.start_radian * 180 / math.pi, i.end_radian * 180 / math.pi, i.color])
                else:
                    map_element.append([i.type, i.init_pos, i.color])
            self.board_width = self.env_core.view_setting['width'] + 2 * self.env_core.view_setting['edge']
            self.board_height = self.env_core.view_setting['height'] + 2 * self.env_core.view_setting['edge']

            self.init_info["board_height"] = self.board_height
            self.init_info["board_width"] = self.board_width
            if self.game_name == 'running':
                self.init_info["map_num"] = rand_map_idx
                self.init_info["map_objects"] = map_element
                self.env_core.map_num = rand_map_idx
            else:
                self.init_info["scenario"] = self.env_core.game_name
                self.init_info["map_objects"] = map_element

        _ = self.reset()

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        if not seed:  # use previous seed when no new seed input
            seed = self.seed
        else:  # update env global seed
            self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, shuffle_map=False):
        self.step_cnt = 0
        self.done = False
        self.won = {}
        self.n_return = [0] * self.n_player
        if self.game_name == 'billiard':
            init_obs = self.env_core.reset()

            self.init_info["agent_position"] = self.env_core.agent_pos
            self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(
                len(self.env_core.agent_list))]  # copy.deepcopy(self.env_core.agent_theta)
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in
                                              range(len(self.env_core.agent_list))]
            self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in
                                           range(len(self.env_core.agent_list))]
            self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in
                                                 range(len(self.env_core.agent_list))]
            self.init_info['agent2idx'] = list(self.env_core.agent2idx.keys())

            self.current_state = init_obs
            self.all_observes = self.get_all_observes()
        elif self.game_name == 'tablehockey':
            init_obs = self.env_core.reset()
            self.ball_pos_init()

            self.init_info["agent_position"] = self.env_core.agent_pos
            self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(
                len(self.env_core.agent_list))]  # copy.deepcopy(self.env_core.agent_theta)
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in
                                              range(len(self.env_core.agent_list))]
            self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in
                                           range(len(self.env_core.agent_list))]
            self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in
                                                 range(len(self.env_core.agent_list))]

            self.current_state = init_obs
            self.all_observes = self.get_all_observes()
            self.ball_end_pos = None

        elif self.game_name == 'football':
            init_obs = self.env_core.reset()
            self.ball_pos_init()

            self.init_info["agent_position"] = self.env_core.agent_pos
            self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(
                len(self.env_core.agent_list))]  # copy.deepcopy(self.env_core.agent_theta)
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in
                                              range(len(self.env_core.agent_list))]
            self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in
                                           range(len(self.env_core.agent_list))]
            self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in
                                                 range(len(self.env_core.agent_list))]

            self.current_state = init_obs
            self.all_observes = self.get_all_observes()
            self.ball_end_pos = None
        elif self.game_name == 'wrestling':
            init_obs = self.env_core.reset()

            self.init_info["agent_position"] = self.env_core.agent_pos
            self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(
                len(self.env_core.agent_list))]  # copy.deepcopy(self.env_core.agent_theta)
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in
                                              range(len(self.env_core.agent_list))]
            self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in
                                           range(len(self.env_core.agent_list))]
            self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in
                                                 range(len(self.env_core.agent_list))]

            self.current_state = init_obs
            self.all_observes = self.get_all_observes()
        elif self.game_name == 'curling':
            self.current_state = self.env_core.reset()
            self.current_game_round = self.env_core.game_round
            self.current_score = [self.env_core.purple_game_point, self.env_core.green_game_point]
            self.current_throws_left = [self.env_core.max_n - self.env_core.num_purple,
                                        self.env_core.max_n - self.env_core.num_green]
            self.current_release = self.env_core.release

            self.init_info["agent_position"] = self.env_core.agent_pos
            self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(
                len(self.env_core.agent_list))]  # copy.deepcopy(self.env_core.agent_theta)
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            # self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in
                                           range(len(self.env_core.agent_list))]
            self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in
                                                 range(len(self.env_core.agent_list))]

            self.all_observes = self.get_all_observes()
        elif self.game_name == 'integrated':
            init_obs = self.env_core.reset()

            map_element = []
            for i in self.env_core.game_pool:
                name, game = i['name'], i['game']
                temp_game_element = []
                for i in game.map['objects']:
                    if i.type == 'arc':
                        temp_game_element.append(
                            [i.type, i.init_pos, i.start_radian * 180 / math.pi, i.end_radian * 180 / math.pi, i.color])
                    else:
                        temp_game_element.append([i.type, i.init_pos, i.color])
                map_element.append({"game name": name, "game map": temp_game_element,
                                    "board_height": game.view_setting['width'] + 2 * game.view_setting['edge'],
                                    "board_width": game.view_setting['height'] + 2 * game.view_setting['edge']})
                if name == 'running-competition':
                    map_element[-1]['running_map'] = self.env_core.running_game.map_index

            self.init_info = {"scenario": "Olympics-integrated", "all_scenario_objects": map_element}
            self.init_info['game_order'] = [self.env_core.game_pool[i]['name'] for i in
                                            self.env_core.selected_game_idx_pool]
            self.init_info['current_game'] = self.env_core.game_name
            self.init_info['agent_position'] = self.env_core.agent_pos
            self.init_info['agent_direction'] = [self.env_core.agent_theta[i][0] for i in
                                                 range(len(self.env_core.agent_list))]
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in
                                              range(len(self.env_core.agent_list))]
            self.init_info["agent_vis"] = [self.env_core.agent_list[i].visibility for i in
                                           range(len(self.env_core.agent_list))]
            self.init_info["agent_vis_clear"] = [self.env_core.agent_list[i].visibility_clear for i in
                                                 range(len(self.env_core.agent_list))]
            self.init_info['agent_view_boundary'] = [self.env_core.obs_boundary[i] for i in
                                                     range(len(self.env_core.agent_list))]

            self.current_state = init_obs
            self.all_observes = self.get_all_observes()
        elif self.game_name == 'running':
            # self.set_seed()

            if shuffle_map:  # if shuffle the map, randomly sample a map again
                map_index_seq = list(range(1, self.num_map + 1))
                rand_map_idx = random.choice(map_index_seq)
                Gamemap = create_scenario("map" + str(rand_map_idx))
                self.env_core = Running_competition(map_id=rand_map_idx)
                self.env_core.map_num = rand_map_idx

            self.env_core.reset()

            self.init_info["agent_position"] = self.env_core.agent_pos
            self.init_info["agent_direction"] = [self.env_core.agent_theta[i][0] for i in range(
                len(self.env_core.agent_list))]  # copy.deepcopy(self.env_core.agent_theta)
            self.init_info["agent_color"] = [self.env_core.agent_list[i].color for i in
                                             range(len(self.env_core.agent_list))]
            self.init_info["agent_r"] = [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))]
            self.init_info["agent_energy"] = [self.env_core.agent_list[i].energy for i in
                                              range(len(self.env_core.agent_list))]

            self.current_state = self.env_core.get_obs()
            self.all_observes = self.get_all_observes()
        return self.all_observes

    def ball_pos_init(self):
        y_min, y_max = 300, 500
        for index, item in enumerate(self.env_core.agent_list):
            if item.type == 'ball':
                random_y = random.uniform(y_min, y_max)
                self.env_core.agent_init_pos[index][1] = random_y

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        joint_action_decode = self.decode(joint_action)
        info_before = {"actions": [i for i in joint_action_decode]}
        if 'curling' in self.env_core.game_name or self.game_name == 'curling':
            info_before['current_team'] = self.env_core.current_team
        if self.env_core.game_name == 'integrated':
            info_before['current_game'] = self.env_core.game_name

        all_observations, reward, done, info_after = self.env_core.step(joint_action_decode)
        if self.game_name == 'running':
            info_after = self.step_before_info()
        else:
            info_after = self.step_after_info()
        self.current_state = all_observations

        if self.game_name == 'curling':
            self.current_game_round = self.env_core.game_round
            self.current_score = [self.env_core.purple_game_point, self.env_core.green_game_point]
            self.current_throws_left = [self.env_core.max_n - self.env_core.num_purple,
                                        self.env_core.max_n - self.env_core.num_green]
            self.current_release = self.env_core.release

        self.all_observes = self.get_all_observes()

        self.step_cnt += 1
        self.done = done
        if self.done:
            if self.game_name == 'football' or self.game_name == 'tablehockey':
                self.ball_position()
            self.set_n_return()

        return self.all_observes, reward, self.done, info_before, info_after

    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:  # check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

        if self.game_name == 'curling':
            for idx, team_action in enumerate(joint_action):
                if not team_action:
                    joint_action[idx] = [[random.randint(-100, 200)], [random.randint(-30, 30)]]

            for idx, team_action in enumerate(joint_action):
                if not (-100 <= team_action[0][0] <= 200) or not (-30 <= team_action[1][0] <= 30):
                    joint_action[idx] = [[random.randint(-100, 200)], [random.randint(-30, 30)]]

            return joint_action
        else:
            for idx, team_action in enumerate(joint_action):
                if not (-100 <= team_action[0][0] <= 200) or not (-30 <= team_action[1][0] <= 30):
                    joint_action[idx] = [[0], [0]]

            return joint_action

    def step_after_info(self, info=''):
        if self.game_name != 'running':
            if self.game_name == 'integrated':
                if self.env_core.game_name == 'billiard':
                    sorted_agent2idx = sorted(self.env_core.agent2idx.items(),
                                              key=lambda x: x[0])  # agent_0, agent_1, ball_0, ...
                    sorted_pos, sorted_direction, sorted_energy, sorted_view_boundary = [], [], [], []
                    for i in sorted_agent2idx:
                        index = i[1]
                        if index is None:  # this ball is deleted
                            sorted_pos.append(None)
                            sorted_direction.append(None)
                            sorted_energy.append(None)
                            sorted_view_boundary.append(None)
                        else:
                            sorted_pos.append(self.env_core.agent_pos[index])
                            sorted_direction.append(self.env_core.agent_theta[index])
                            sorted_energy.append(self.env_core.agent_list[index].energy)
                            sorted_view_boundary.append(self.env_core.obs_boundary[index])
                    info = {
                        "current_game": self.env_core.game_name, "agent_position": sorted_pos,
                        "agent_direction": sorted_direction, "agent_energy": sorted_energy,
                        "game_score": [self.env_core.game_score[i] for i in range(2)],
                        "agent_view_boundary": sorted_view_boundary,
                        "agent_r": [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))],
                    }

                elif 'curling' in self.env_core.game_name:
                    info = {'current_game': self.env_core.game_name, "agent_position": self.env_core.agent_pos,
                            "agent_direction": [self.env_core.agent_theta[i][0] for i in range(len(self.env_core.agent_list))],
                            "agent_energy": [self.env_core.agent_list[i].energy for i in range(len(self.env_core.agent_list))],
                            "game_score": [self.env_core.game_score[i] for i in range(2)],
                            "agent_view_boundary": [self.env_core.obs_boundary[i] for i in
                                                    range(len(self.env_core.agent_list))],
                            "agent_color": [self.env_core.agent_list[i].color for i in range(len(self.env_core.agent_list))],
                            "agent_r": [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))],
                            "current_round": self.env_core.game_round
                            }

                else:
                    info = {'current_game': self.env_core.game_name, "agent_position": self.env_core.agent_pos,
                            "agent_direction": [self.env_core.agent_theta[i][0] for i in range(len(self.env_core.agent_list))],
                            "agent_energy": [self.env_core.agent_list[i].energy for i in range(len(self.env_core.agent_list))],
                            "game_score": [self.env_core.game_score[i] for i in range(2)],
                            "agent_view_boundary": [self.env_core.obs_boundary[i] for i in
                                                    range(len(self.env_core.agent_list))]}
            elif self.game_name == 'billiard':
                sorted_agent2idx = sorted(self.env_core.agent2idx.items(),
                                          key=lambda x: x[0])
                sorted_pos, sorted_direction, sorted_energy = [], [], []
                for i in sorted_agent2idx:
                    index = i[1]
                    if index is None:  # this ball is deleted
                        sorted_pos.append(None)
                        sorted_direction.append(None)
                        sorted_energy.append(None)
                    else:
                        sorted_pos.append(self.env_core.agent_pos[index])
                        sorted_direction.append(self.env_core.agent_theta[index])
                        sorted_energy.append(self.env_core.agent_list[index].energy)

                score = [self.env_core.total_score[i] for i in range(2)]

                info = {'agent_position': sorted_pos, 'agent_direction': sorted_direction,
                        'agent_energy': sorted_energy, 'score': score}
            elif self.game_name == 'curling':
                info = {"agent_position": self.env_core.agent_pos,
                        "agent_direction": [self.env_core.agent_theta[i][0] for i in
                                            range(len(self.env_core.agent_list))],
                        "agent_color": [self.env_core.agent_list[i].color for i in
                                        range(len(self.env_core.agent_list))],
                        "agent_r": [self.env_core.agent_list[i].r for i in range(len(self.env_core.agent_list))],
                        "current_game": self.env_core.game_round,
                        "score": [self.env_core.purple_game_point, self.env_core.green_game_point]}
            else:
                info = {"agent_position": self.env_core.agent_pos,
                        "agent_direction": [self.env_core.agent_theta[i][0] for i in
                                            range(len(self.env_core.agent_list))],
                        "agent_energy": [self.env_core.agent_list[i].energy for i in
                                         range(len(self.env_core.agent_list))]}

            return info

    def step_before_info(self, info=''):

        info = {"agent_position":self.env_core.agent_pos, "agent_direction":[self.env_core.agent_theta[i][0] for i in range(len(self.env_core.agent_list))],
                "agent_energy":[self.env_core.agent_list[i].energy for i in range(len(self.env_core.agent_list))]}
        return info

    def decode(self, joint_action):
        joint_action_decode = []
        for act_id, nested_action in enumerate(joint_action):
            temp_action = [0, 0]
            temp_action[0] = nested_action[0][0]
            temp_action[1] = nested_action[1][0]
            joint_action_decode.append(temp_action)

        return joint_action_decode

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            if self.game_name == 'curling':
                each = {"obs": self.current_state[i], 'team color': ['purple', 'green'][i],
                        'release': self.current_release,
                        'game round': self.current_game_round, "throws left": self.current_throws_left,
                        "score": self.current_score, "controlled_player_index": i}
            else:
                each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)

        return all_observes

    def set_action_space(self):
        return [[Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))] for _ in range(self.n_player)]

    def get_reward(self, reward):
        return [reward]

    def is_terminal(self):
        if self.game_name == 'billiard' or self.game_name == 'curling':
            return self.done
        elif self.game_name == 'running':
            if self.step_cnt >= self.max_step:
                return True
            for agent_idx in range(self.n_player):
                if self.env_core.agent_list[agent_idx].finished:
                    return True
            return False
        else:
            return self.done

    def ball_position(self):
        self.ball_end_pos = None
        for agent_idx in range(self.env_core.agent_num):
            agent = self.env_core.agent_list[agent_idx]
            if agent.type == 'ball' and agent.finished:
                self.ball_end_pos = self.env_core.agent_pos[agent_idx]

    def set_n_return(self):
        if self.game_name == 'billiard':
            total_reward = self.env_core.total_score
            if total_reward[0] > total_reward[1]:
                self.n_return = [1, 0.]
            elif total_reward[0] < total_reward[1]:
                self.n_return = [0., 1]
            else:
                self.n_return = [0., 0]
        elif self.game_name == 'tablehockey':
            if self.ball_end_pos is None:
                self.n_return = [0, 0]
            else:
                if self.ball_end_pos[0] < 400:
                    if self.env_core.agent_pos[0][0] < 400:
                        self.n_return = [0, 1]
                    else:
                        self.n_return = [1, 0]
                elif self.ball_end_pos[0] > 400:
                    if self.env_core.agent_pos[0][0] < 400:
                        self.n_return = [1, 0]
                    else:
                        self.n_return = [0, 1]
                else:
                    raise NotImplementedError
        elif self.game_name == 'football':
            if self.ball_end_pos is None:
                self.n_return = [0, 0]
            else:
                if self.ball_end_pos[0] < 400:
                    self.n_return = [0, 1]
                elif self.ball_end_pos[0] > 400:
                    self.n_return = [1, 0]
                else:
                    raise NotImplementedError
        elif self.game_name == 'wrestling':
            if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
                self.n_return = [0, 1]
            elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
                self.n_return = [1, 0]
            elif self.env_core.agent_list[0].finished and self.env_core.agent_list[1].finished:
                self.n_return = [0, 0]
            else:
                self.n_return = [0, 0]
        elif self.game_name == 'curling':
            winner = self.env_core.final_winner
            if winner == 0:
                self.n_return = [1, 0.]
            elif winner == 1:
                self.n_return = [0., 1]
            elif winner == -1:
                self.n_return = [0., 0]
            else:
                raise NotImplementedError
        elif self.game_name == 'integrated':
            final_reward = self.env_core.final_reward

            if final_reward[0] > final_reward[1]:
                self.n_return = [1, 0]
            elif final_reward[1] > final_reward[0]:
                self.n_return = [0, 1]
            else:
                self.n_return = [0, 0]
        elif self.game_name == 'running':
            if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
                self.n_return = [1, 0]
            elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
                self.n_return = [0, 1]
            elif self.env_core.agent_list[0].finished and self.env_core.agent_list[1].finished:
                self.n_return = [1, 1]
            else:
                self.n_return = [0, 0]

    def check_win(self):
        if self.game_name == 'billiard':
            total_reward = self.env_core.total_score
            if total_reward[0] > total_reward[1]:
                return '0'
            elif total_reward[0] < total_reward[1]:
                return '1'
            else:
                return '-1'
        elif self.game_name == 'tablehockey':
            if self.ball_end_pos is None:
                return '-1'
            else:
                if self.ball_end_pos[0] < 400:
                    if self.env_core.agent_pos[0][0] < 400:
                        return '1'
                    else:
                        return '0'
                elif self.ball_end_pos[0] > 400:
                    if self.env_core.agent_pos[0][0] < 400:
                        return '0'
                    else:
                        return '1'
                else:
                    raise NotImplementedError
        elif self.game_name == 'football':
            if self.ball_end_pos is None:
                return '-1'
            else:
                if self.ball_end_pos[0] < 400:
                    return '1'
                elif self.ball_end_pos[0] > 400:
                    return '0'
                else:
                    raise NotImplementedError
        elif self.game_name == 'wrestling':
            if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
                return '1'
            elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
                return '0'
            else:
                return '-1'
        elif self.game_name == 'curling':
            winner = self.env_core.final_winner
            return str(winner)
        elif self.game_name == 'integrated':
            final_reward = self.env_core.final_reward

            if final_reward[0] > final_reward[1]:
                return '0'
            elif final_reward[1] > final_reward[0]:
                return '1'
            else:
                return '-1'
        elif self.game_name == 'running':
            if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
                return '0'
            elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
                return '1'
            else:
                return '-1'

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def specify_a_map(self, num):
        assert num <= self.num_map, print('the num is larger than the total number of map')
        Gamemap = create_scenario("map"+str(num))
        self.env_core = Running_competition(map_id = num)
        _ = self.reset()
        self.env_core.map_num = num
