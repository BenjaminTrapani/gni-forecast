import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 70)
with open('data/nba_basketball_data.csv') as player_file:
    with open('data/season_logs.csv') as team_file:
        # players will be listed in at most one game
        player_df = pd.read_csv(player_file)
        player_df['played'] = player_df['fgm'].notnull()
        player_df.drop(['team_abbreviation', 'team_city', 'comment',
                        'fg_pct', 'fg3_pct', 'ft_pct', 'efg_pct', 'off_rating',
                        'def_rating', 'net_rating', 'ast_tov', 'ast_ratio', 'opp'], inplace=True, axis=1)
        start_pos_dummies = pd.get_dummies(player_df['start_position'])
        start_pos_dummies.rename(columns=lambda col: 'start_position_' + str(col), inplace=True)
        player_df = pd.concat([player_df, start_pos_dummies], axis=1)
        player_df.drop('start_position', inplace=True, axis=1)
        player_df['start'] = player_df['start'].apply(lambda start: 1 if start else 0)
        # there will be two entries with the same game ID, one per team. All stats will be
        # in terms of 'team_id'
        team_df = pd.read_csv(team_file)
        team_df.drop(['season_id', 'team_name', 'fg_pct', 'fg3_pct', 'ft_pct', 'plus_minus',
                      'video_available', 'opp'], axis=1, inplace=True)
        team_df['wl'] = team_df['wl'].apply(lambda wl: 1 if wl == 'W' else 0)
        team_df['home'] = team_df['home'].apply(lambda home: 1 if home else 0)

        # Should have two entries per game: one with data on the first team, one with data on the second team
        # Joined result rows will only contain team data + 1 player on the shared 'team_id'
        #merged_data = pd.merge(player_df, team_df, on=['game_id', 'team_id'], how='inner',
                               #suffixes=['_player', '_team'])

        # convert team data + 1 player on same team to format team data + all player data on same team
        distinct_team_game_pairs = team_df[['game_id', 'team_id']].drop_duplicates()
        result_df = None
        team_ID_to_last_game_ID = {}
        for team_game_pair in distinct_team_game_pairs.iterrows():
            cur_game_id = team_game_pair[1]['game_id']
            cur_team_id = team_game_pair[1]['team_id']
            if cur_team_id in team_ID_to_last_game_ID:
                last_game_id = team_ID_to_last_game_ID[cur_team_id]
                left_team_and_players = fetch_team_and_players(team_df, player_df, cur_team_id, last_game_id)
                left_team_and_players.add_suffix('_left')
                remaining_team_id = team_df[team_df['game_id'] == last_game_id and
                                        team_df['team_id'] is not cur_team_id]['team_id']
                right_team_and_players = fetch_team_and_players(team_df, player_df, remaining_team_id, last_game_id)
                right_team_and_players.add_suffix('_right')

                cur_pred_row = pd.concat([left_team_and_players, right_team_and_players], axis=1)

                if result_df is None:
                    result_df = pd.DataFrame(columns=cur_pred_row.columns.values)

                result_df = pd.Concat([result_df, cur_pred_row])
            team_ID_to_last_game_ID[cur_team_id] = cur_game_id

        print(result_df)
        #merged_data.to_csv('data/merged-preprocessed.csv')
