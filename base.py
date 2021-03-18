# 定义全局函数便于直接调用处理
from datetime import datetime
from sklearn import metrics
import numpy as np
import pandas as pd
from chinese_calendar import is_in_lieu, is_holiday, is_workday
from scipy.stats import boxcox

############全局参数#################################
discrete_list = ['pax_name_passport', 'seg_route_from', 'seg_route_to', 'seg_flight', 'seg_cabin',
                 'seg_dep_time', 'gender', 'age', 'birth_date', 'residence_country', 'nation_name',
                 'city_name',
                 'province_name', 'marital_stat', 'ffp_nbr', 'member_level', 'often_city', 'enroll_chnl',
                 'pref_aircraft_m3_1', 'pref_aircraft_m3_2', 'pref_aircraft_m3_3', 'pref_aircraft_m3_4',
                 'pref_aircraft_m3_5', 'pref_aircraft_m6_1', 'pref_aircraft_m6_2', 'pref_aircraft_m6_3',
                 'pref_aircraft_m6_4', 'pref_aircraft_m6_5', 'pref_aircraft_y1_1', 'pref_aircraft_y1_2',
                 'pref_aircraft_y1_3', 'pref_aircraft_y1_4', 'pref_aircraft_y1_5', 'pref_aircraft_y2_1',
                 'pref_aircraft_y2_2', 'pref_aircraft_y2_3', 'pref_aircraft_y2_4', 'pref_aircraft_y2_5',
                 'pref_aircraft_y3_1', 'pref_aircraft_y3_2', 'pref_aircraft_y3_3', 'pref_aircraft_y3_4',
                 'pref_aircraft_y3_5', 'pref_orig_m3_1', 'pref_orig_m3_2', 'pref_orig_m3_3', 'pref_orig_m3_4',
                 'pref_orig_m3_5', 'pref_orig_m6_1', 'pref_orig_m6_2', 'pref_orig_m6_3', 'pref_orig_m6_4',
                 'pref_orig_m6_5', 'pref_orig_y1_1', 'pref_orig_y1_2', 'pref_orig_y1_3', 'pref_orig_y1_4',
                 'pref_orig_y1_5', 'pref_orig_y2_1', 'pref_orig_y2_2', 'pref_orig_y2_3', 'pref_orig_y2_4',
                 'pref_orig_y2_5', 'pref_orig_y3_1', 'pref_orig_y3_2', 'pref_orig_y3_3', 'pref_orig_y3_4',
                 'pref_orig_y3_5', 'pref_line_m3_1', 'pref_line_m3_2', 'pref_line_m3_3', 'pref_line_m3_4',
                 'pref_line_m3_5', 'pref_line_m6_1', 'pref_line_m6_2', 'pref_line_m6_3', 'pref_line_m6_4',
                 'pref_line_m6_5', 'pref_line_y1_1', 'pref_line_y1_2', 'pref_line_y1_3', 'pref_line_y1_4',
                 'pref_line_y1_5', 'pref_line_y2_1', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y2_4',
                 'pref_line_y2_5', 'pref_line_y3_1', 'pref_line_y3_2', 'pref_line_y3_3', 'pref_line_y3_4',
                 'pref_line_y3_5', 'pref_city_m3_1', 'pref_city_m3_2', 'pref_city_m3_3', 'pref_city_m3_4',
                 'pref_city_m3_5', 'pref_city_m6_1', 'pref_city_m6_2', 'pref_city_m6_3', 'pref_city_m6_4',
                 'pref_city_m6_5', 'pref_city_y1_1', 'pref_city_y1_2', 'pref_city_y1_3', 'pref_city_y1_4',
                 'pref_city_y1_5', 'pref_city_y2_1', 'pref_city_y2_2', 'pref_city_y2_3', 'pref_city_y2_4',
                 'pref_city_y2_5', 'pref_city_y3_1', 'pref_city_y3_2', 'pref_city_y3_3', 'pref_city_y3_4',
                 'pref_city_y3_5', 'recent_flt_day', 'pit_add_chnl_m3', 'pit_add_chnl_m6', 'pit_add_chnl_y1',
                 'pit_add_chnl_y2', 'pit_add_chnl_y3', 'pref_orig_city_m3', 'pref_orig_city_m6', 'pref_orig_city_y1',
                 'pref_orig_city_y2', 'pref_orig_city_y3', 'pref_dest_city_m3', 'pref_dest_city_m6',
                 'pref_dest_city_y1', 'pref_dest_city_y2', 'pref_dest_city_y3',
                 # 对月份，小时等新增为离散类型数据进行分析，这里一定要详细判断一下数据的分布情况
                 'pref_month_m3_1', 'pref_month_m3_2', 'pref_month_m3_3', 'pref_month_m3_4', 'pref_month_y3_5',
                 'pref_month_m3_5', 'pref_month_m6_1', 'pref_month_m6_2', 'pref_month_m6_3', 'pref_month_m6_4',
                 'pref_month_m6_5', 'pref_month_y1_1', 'pref_month_y1_2', 'pref_month_y1_3', 'pref_month_y1_4',
                 'pref_month_y1_5', 'pref_month_y2_1', 'pref_month_y2_2', 'pref_month_y2_3', 'pref_month_y2_4',
                 'pref_month_y2_5', 'pref_month_y3_1', 'pref_month_y3_2', 'pref_month_y3_3', 'pref_month_y3_4',
                 'seg_dep_time_month', 'seg_dep_time_year', 'seg_dep_time_hour', 'seg_dep_time_is_workday',
                 'seg_dep_time_is_holiday', 'seg_dep_time_is_in_lieu', 'recent_flt_day_month', 'recent_flt_day_year',
                 'recent_flt_day_hour', 'recent_flt_day_is_workday', 'recent_flt_day_is_holiday',
                 'recent_flt_day_is_in_lieu', 'has_ffp_nbr'
                 ]  # 离散型的特征
epsilon = 1e-5
func_dict = {'add': lambda x, y: x + y,
             "mins": lambda x, y: x - y,
             'div': lambda x, y: x / (y + epsilon),
             'multi': lambda x, y: x + y}
func_list = ['mean', 'median', 'min', 'max', 'std', 'var']
drop_features = ['emd_lable', 'cabin_hf_cnt_m3', 'cabin_hf_cnt_m6', 'complain_valid_cnt_m3', 'complain_valid_cnt_m6',
                 'bag_cnt_m6',
                 'bag_cnt_y1', 'bag_cnt_y2', 'bag_cnt_y3', 'flt_cancel_cnt_y3', 'cabin_fall_cnt_y3', 'complain_cnt_m3',
                 'complain_cnt_m6', 'pit_out_amt',
                 'pit_add_non_amt_m3', 'pit_add_buy_amt_y2', 'pit_add_buy_amt_y3', 'pit_des_out_amt_m3',
                 'pit_des_out_amt_m6', 'pit_des_out_amt_y1', 'pit_add_non_cnt_m3', 'pit_add_buy_cnt_y2',
                 'pit_add_buy_cnt_y3', 'pit_des_mall_cnt_m6', 'pit_des_mall_cnt_y1', 'pit_des_out_cnt_m6',
                 'pit_des_out_cnt_y1', 'pit_ech_avg_amt_m3',
                 'pit_out_avg_amt_m3', 'pit_out_avg_amt_m6']  # 观察分布后预筛选的特征,都是连续性的
train_drop_features = ['pax_name_passport', 'seg_dep_time', 'recent_flt_day',
                       'birth_date', 'ffp_nbr']  # 训练过程中需要剔除的变量，如用户ID、日期等明显的无关数据！
feature_subset_list = [['dist_cnt_y3',
                        'tkt_all_amt_y3',
                        'pref_line_y3_1',
                        'tkt_3y_amt',
                        'pref_line_y3_2',
                        'pref_line_y3_4',
                        'pref_city_y3_3',
                        'dist_i_cnt_y2',
                        'tkt_avg_amt_y1',
                        'select_seat_cnt_y3',
                        'dist_cnt_y1',
                        'pax_tax',
                        'tkt_d_amt_y3',
                        'dist_all_cnt_y2',
                        'dist_i_cnt_y3',
                        'tkt_all_amt_y2',
                        'tkt_i_amt_y1',
                        'pax_fcny',
                        'tkt_avg_amt_y2',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'pref_line_y2_3',
                        'flt_delay_time_y1',
                        'pref_line_y2_1',
                        'tkt_avg_amt_y3',
                        'dist_d_cnt_y3',
                        'flt_delay_time_y2',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'tkt_i_amt_y3'],
                       ['tkt_d_amt_y1',
                        'pref_city_y2_4',
                        'pref_orig_y3_2',
                        'avg_dist_cnt_m6',
                        'dist_cnt_y3',
                        'pref_city_y1_4',
                        'pref_city_y2_3',
                        'pref_line_m6_2',
                        'pref_line_m3_4',
                        'pref_orig_y2_4',
                        'pref_line_y3_3',
                        'tkt_3y_amt',
                        'pref_orig_y1_5',
                        'pref_line_y2_4',
                        'tkt_avg_amt_y1',
                        'pref_line_y2_5',
                        'tkt_i_amt_y2',
                        'pref_city_y1_2',
                        'pax_tax',
                        'avg_dist_cnt_y1',
                        'pref_orig_y3_4',
                        'pref_aircraft_y1_5',
                        'tkt_d_amt_y3',
                        'dist_all_cnt_y2',
                        'tkt_all_amt_y2',
                        'tkt_i_amt_y1',
                        'pref_orig_y2_2',
                        'pax_fcny',
                        'tkt_avg_amt_y2',
                        'pref_line_m3_3',
                        'flt_delay_time_m6',
                        'seg_cabin',
                        'avg_dist_cnt_y3',
                        'pref_line_y2_3',
                        'pref_line_y2_1',
                        'tkt_avg_amt_y3',
                        'pit_avg_amt_y3',
                        'flt_delay_time_y1',
                        'pit_accu_amt_y3',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'pref_city_y3_5',
                        'dist_all_cnt_y3',
                        'pref_aircraft_y2_4',
                        'tkt_i_amt_y3'],
                       ['tkt_all_amt_m6',
                        'avg_dist_cnt_m6',
                        'flt_delay_time_m3',
                        'dist_cnt_y3',
                        'tkt_all_amt_y3',
                        'tkt_3y_amt',
                        'pref_line_y3_3',
                        'seg_route_to',
                        'pref_line_y3_4',
                        'pref_line_y2_4',
                        'tkt_i_amt_y3',
                        'pref_aircraft_y3_3',
                        'dist_i_cnt_y2',
                        'tkt_avg_amt_y1',
                        'select_seat_cnt_y3',
                        'dist_cnt_y1',
                        'pref_line_y2_5',
                        'tkt_i_amt_y2',
                        'pax_tax',
                        'avg_dist_cnt_y1',
                        'pref_aircraft_y2_5',
                        'tkt_d_amt_y3',
                        'dist_all_cnt_y2',
                        'pref_line_m3_2',
                        'dist_i_cnt_y3',
                        'pit_add_air_amt_y3',
                        'tkt_all_amt_y2',
                        'tkt_i_amt_y1',
                        'pref_line_y1_2',
                        'flt_leg_cnt_y1',
                        'pit_pay_avg_amt_y3',
                        'tkt_all_amt_y1',
                        'pax_fcny',
                        'tkt_avg_amt_m6',
                        'tkt_i_amt_m6',
                        'flt_nature_cnt_y3',
                        'pref_line_y3_5',
                        'flt_delay_time_m6',
                        'seg_cabin',
                        'avg_dist_cnt_y3',
                        'pref_line_y2_1',
                        'flt_delay_time_y1',
                        'flt_delay_time_y2',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'seg_dep_time_hour',
                        'dist_all_cnt_y1'],
                       ['tkt_d_amt_y1',
                        'tkt_all_amt_m6',
                        'pref_city_y2_2',
                        'pref_city_y2_4',
                        'pref_orig_y3_2',
                        'flt_delay_time_m3',
                        'tkt_all_amt_y3',
                        'dist_cnt_y3',
                        'pit_cons_amt_y2',
                        'pref_city_y2_3',
                        'pit_des_out_amt_y2',
                        'pref_city_y2_5',
                        'tkt_3y_amt',
                        'pref_line_y2_4',
                        'pit_avg_cons_amt_y3',
                        'pref_orig_y2_3',
                        'pref_aircraft_y3_3',
                        'pref_city_y3_4',
                        'dist_i_cnt_y2',
                        'pref_line_m6_3',
                        'tkt_avg_amt_y1',
                        'pref_aircraft_y3_4',
                        'tkt_i_amt_y2',
                        'pax_tax',
                        'tkt_d_amt_y3',
                        'dist_all_cnt_y2',
                        'pit_pay_avg_amt_y2',
                        'pref_line_m3_2',
                        'pit_add_air_amt_y1',
                        'dist_i_cnt_y3',
                        'seg_dep_time_month',
                        'dist_i_cnt_y1',
                        'pref_aircraft_y3_5',
                        'pit_accu_air_amt',
                        'pit_all_amt',
                        'pref_line_y1_2',
                        'pit_pay_avg_amt_y3',
                        'tkt_all_amt_y1',
                        'tkt_i_amt_y1',
                        'pax_fcny',
                        'tkt_avg_amt_y2',
                        'tkt_all_amt_y2',
                        'pref_orig_y2_2',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'pref_orig_y3_5',
                        'pref_line_y2_3',
                        'flt_delay_time_y2',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'dist_all_cnt_y1'],
                       ['pref_dest_city_y1',
                        'tkt_d_amt_y2',
                        'flt_delay_time_m3',
                        'tkt_all_amt_y3',
                        'pref_orig_m6_5',
                        'pref_city_y2_5',
                        'tkt_3y_amt',
                        'pref_line_y3_3',
                        'pit_accu_amt_y2',
                        'avg_dist_cnt_y2',
                        'seg_dep_time_hour',
                        'pref_city_y3_3',
                        'dist_i_cnt_y2',
                        'dist_cnt_y1',
                        'pit_income_avg_amt_y3',
                        'pref_line_y2_5',
                        'pref_city_y1_2',
                        'avg_dist_cnt_y1',
                        'pax_tax',
                        'pref_orig_y3_4',
                        'pref_line_y1_3',
                        'tkt_d_amt_y3',
                        'dist_i_cnt_y3',
                        'tkt_all_amt_y2',
                        'pax_fcny',
                        'pref_orig_city_m3',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'pref_orig_city_y1',
                        'flt_delay_time_m6',
                        'pref_line_y2_3',
                        'flt_delay_time_y1',
                        'pref_aircraft_y2_4',
                        'tkt_avg_amt_y3',
                        'pit_income_cnt_y2',
                        'pit_accu_amt_y3',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'pref_orig_city_y3'],
                       ['tkt_d_amt_y2',
                        'flt_delay_time_m3',
                        'dist_cnt_y3',
                        'tkt_all_amt_y3',
                        'pref_line_y3_1',
                        'pref_city_y2_5',
                        'pref_line_y3_3',
                        'pref_orig_y2_4',
                        'pref_orig_y3_3',
                        'pref_line_m6_3',
                        'dist_cnt_y1',
                        'pref_line_y2_5',
                        'tkt_i_amt_y2',
                        'pax_tax',
                        'avg_dist_cnt_y1',
                        'tkt_d_amt_y3',
                        'dist_all_cnt_y2',
                        'pit_avg_cons_amt_y2',
                        'dist_i_cnt_y3',
                        'pit_add_air_amt_y3',
                        'tkt_all_amt_y2',
                        'pref_line_y1_2',
                        'pax_fcny',
                        'tkt_avg_amt_y2',
                        'pit_cons_amt_y3',
                        'tkt_avg_amt_m6',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'flt_delay_time_m6',
                        'avg_dist_cnt_y3',
                        'pref_line_y2_1',
                        'flt_delay_time_y1',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'pref_orig_city_y3',
                        'tkt_i_amt_y3'],
                       ['pref_orig_m6_1',
                        'tkt_d_amt_y2',
                        'pref_city_y2_2',
                        'pit_avg_amt_y2',
                        'tkt_all_amt_y3',
                        'dist_cnt_y3',
                        'pref_line_y3_1',
                        'pref_city_y2_3',
                        'tkt_3y_amt',
                        'avg_dist_cnt_y2',
                        'pit_accu_amt_y2',
                        'pref_orig_y3_3',
                        'pref_orig_y1_5',
                        'pref_line_y2_4',
                        'pref_city_y3_3',
                        'dist_i_cnt_y2',
                        'tkt_avg_amt_y1',
                        'dist_cnt_y1',
                        'pref_city_y3_1',
                        'pref_aircraft_m6_4',
                        'tkt_i_amt_y2',
                        'pax_tax',
                        'avg_dist_cnt_y1',
                        'tkt_d_amt_y3',
                        'dist_i_cnt_y3',
                        'dist_i_cnt_y1',
                        'tkt_all_amt_y2',
                        'pref_line_y1_2',
                        'tkt_all_amt_y1',
                        'pref_orig_y2_2',
                        'tkt_avg_amt_y2',
                        'pax_fcny',
                        'pref_orig_city_y2',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'avg_dist_cnt_y3',
                        'tkt_avg_amt_y3',
                        'flt_delay_time_y1',
                        'flt_delay_time_y2',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'seg_dep_time_hour'],
                       ['pref_city_y3_2',
                        'pref_city_y2_2',
                        'flt_delay_time_m3',
                        'tkt_all_amt_y3',
                        'pit_accu_amt_y1',
                        'avg_dist_cnt_y2',
                        'dist_all_cnt_y1',
                        'pref_line_y3_4',
                        'pref_city_y3_3',
                        'pit_avg_interval_y3',
                        'tkt_avg_amt_y1',
                        'dist_cnt_y1',
                        'pit_income_avg_amt_y3',
                        'tkt_i_amt_y2',
                        'pax_tax',
                        'avg_dist_cnt_y1',
                        'pref_dest_city_y2',
                        'dist_all_cnt_y2',
                        'tkt_d_amt_y3',
                        'pit_pay_avg_amt_y2',
                        'dist_i_cnt_y3',
                        'dist_i_cnt_y1',
                        'tkt_all_amt_y2',
                        'pit_all_amt',
                        'pit_accu_air_amt',
                        'pit_pay_avg_amt_y3',
                        'tkt_all_amt_y1',
                        'pref_orig_y2_2',
                        'pax_fcny',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'avg_dist_cnt_y3',
                        'pref_line_y2_1',
                        'pref_line_y2_3',
                        'flt_delay_time_y1',
                        'flt_delay_time_y2',
                        'pit_accu_amt_y3',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'pref_orig_city_y3',
                        'tkt_i_amt_y3'],
                       ['tkt_d_amt_y2',
                        'dist_cnt_y3',
                        'pit_cons_amt_y2',
                        'tkt_all_amt_y3',
                        'pref_city_y2_5',
                        'pref_line_y3_3',
                        'pit_accu_amt_y2',
                        'seg_route_to',
                        'pref_line_y2_4',
                        'flt_leg_i_cnt_y2',
                        'dist_i_cnt_y2',
                        'tkt_avg_amt_y1',
                        'city_name',
                        'select_seat_cnt_y3',
                        'pref_city_y3_1',
                        'tkt_i_amt_y2',
                        'avg_dist_cnt_y1',
                        'pax_tax',
                        'pref_line_m6_1',
                        'pref_aircraft_y2_5',
                        'pit_now_cons_amt',
                        'tkt_d_amt_y3',
                        'pit_avg_cons_amt_y2',
                        'dist_i_cnt_y3',
                        'pit_add_air_amt_y3',
                        'tkt_all_amt_y2',
                        'pref_line_y1_2',
                        'tkt_all_amt_y1',
                        'pax_fcny',
                        'pref_orig_m3_4',
                        'dist_d_cnt_y2',
                        'pref_line_y3_5',
                        'seg_cabin',
                        'flt_delay_time_m6',
                        'avg_dist_cnt_y3',
                        'pref_line_y2_1',
                        'pref_line_y2_3',
                        'pref_orig_y1_1',
                        'seg_flight',
                        'pref_line_y2_2',
                        'dist_all_cnt_y3',
                        'pref_orig_city_y3',
                        'tkt_i_amt_y3'],
                       ['pref_city_y3_2',
                        'tkt_d_amt_y2',
                        'pref_orig_y1_4',
                        'pref_orig_y3_2',
                        'dist_cnt_y3',
                        'tkt_all_amt_y3',
                        'pit_accu_amt_y1',
                        'pref_city_y2_3',
                        'pref_line_y3_1',
                        'pref_line_y3_3',
                        'pref_line_y3_4',
                        'pref_line_y2_4',
                        'tkt_i_amt_y3',
                        'pref_city_y3_3',
                        'tkt_avg_amt_y1',
                        'dist_cnt_y1',
                        'pref_line_y2_5',
                        'tkt_i_amt_y2',
                        'pax_tax',
                        'pref_aircraft_y2_5',
                        'pit_pay_avg_amt_y2',
                        'dist_i_cnt_y3',
                        'pit_add_air_amt_y3',
                        'tkt_all_amt_y2',
                        'pref_line_y1_2',
                        'pit_accu_air_amt',
                        'cabin_hi_cnt_y2',
                        'pax_fcny',
                        'tkt_avg_amt_y2',
                        'tkt_avg_amt_m6',
                        'seg_cabin',
                        'pref_line_y2_3',
                        'tkt_avg_amt_y3',
                        'pit_avg_amt_y3',
                        'pit_accu_amt_y3',
                        'select_seat_cnt_y2',
                        'seg_flight',
                        'pref_line_y2_2',
                        'pref_dest_city_m6',
                        'dist_all_cnt_y3',
                        'seg_dep_time_hour',
                        'dist_all_cnt_y1']]
############目录定义#################################
datapath = 'D:/outsourcing/data/'
featurepath = 'D:/outsourcing/feature/'
resultpath = 'D:/outsourcing/result/'
tmppath = 'D:/outsourcing/tmp/'


###############函数定义################################

def read_csv(file_name, num_rows=None):
    if num_rows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=num_rows)


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def evaluation(x_train, x_test, y_train, y_test, str):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    # knn = KNeighborsClassifier(n_neighbors=5, p=1)
    bayes = GaussianNB()
    tree = DecisionTreeClassifier()
    svm = SVC()
    LR = LogisticRegression()
    model_list = [bayes, tree, svm, LR]
    model_name = ['bayes', 'tree', 'svm', 'LR']
    f = open('../result/' + str + '.txt', mode='x')
    for i in range(len(model_list)):
        np.random.seed(0)
        model = model_list[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred2 = model.predict(x_train)
        print("###################" + model_name[i] + "#########################", file=f)
        from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, \
            accuracy_score, roc_auc_score, confusion_matrix
        print("balanced_accuracy_score=", balanced_accuracy_score(y_pred=y_pred, y_true=y_test),
              balanced_accuracy_score(y_pred=y_pred2, y_true=y_train), file=f)
        print("f1=", f1_score(y_pred=y_pred, y_true=y_test), f1_score(y_pred=y_pred2, y_true=y_train), file=f)
        print("precision_score=", precision_score(y_pred=y_pred, y_true=y_test),
              precision_score(y_pred=y_pred2, y_true=y_train), file=f)
        print("recall_score=", recall_score(y_pred=y_pred, y_true=y_test), recall_score(y_pred=y_pred2, y_true=y_train),
              file=f)
        print("accuracy=", accuracy_score(y_pred=y_pred, y_true=y_test), accuracy_score(y_pred=y_pred2, y_true=y_train),
              file=f)
        print("auc=", roc_auc_score(y_true=y_test, y_score=y_pred), roc_auc_score(y_true=y_train, y_score=y_pred2),
              file=f)
        print("#####混淆矩阵#########", file=f)
        print(confusion_matrix(y_true=y_test, y_pred=y_pred), confusion_matrix(y_true=y_train, y_pred=y_pred2), file=f)
    return


def date_transfer(data):
    # 缺失值用-1表示即可
    data['seg_dep_time_month'] = [datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').month for i in
                                  range(len(data))]
    data['seg_dep_time_year'] = [datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').year for i in
                                 range(len(data))]
    data['seg_dep_time_hour'] = [datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').hour for i in
                                 range(len(data))]
    data['seg_dep_time_is_workday'] = [
        is_workday(datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M'))
        for i in
        range(len(data))]
    data['seg_dep_time_is_holiday'] = [
        is_holiday(datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M'))
        for i in
        range(len(data))]
    data['seg_dep_time_is_in_lieu'] = [
        is_in_lieu(datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M'))
        for i in
        range(len(data))]
    data['recent_flt_day_month'] = [
        datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M').month if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_year'] = [
        datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M').year if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_hour'] = [
        datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M').hour if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]

    data['recent_flt_day_is_workday'] = [
        is_workday(datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M')) if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_is_holiday'] = [
        is_holiday(datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M')) if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['recent_flt_day_is_in_lieu'] = [
        is_in_lieu(datetime.strptime(data['recent_flt_day'][i], '%Y/%m/%d %H:%M')) if data['recent_flt_day'][i] != '0'
        else -1
        for i in
        range(len(data))]
    data['birth_interval_day'] = [
        datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').day - datetime.strptime(data['birth_date'][i],
                                                                                             '%Y/%m/%d %H:%M').day
        if data['birth_date'][i] != '0'
        else -999
        for i in
        range(len(data))]
    data['last_flt_day'] = [
        datetime.strptime(data['seg_dep_time'][i], '%Y/%m/%d %H:%M').day - datetime.strptime(data['recent_flt_day'][i],
                                                                                             '%Y/%m/%d %H:%M').day
        if data['recent_flt_day'][i] != '0'
        else -999
        for i in
        range(len(data))]
    data['has_ffp_nbr'] = [0 if data['ffp_nbr'][i] == 0
                           else 1
                           for i in range(len(data))]
    data = data.replace({True: 1, False: 0})
    return data


def Box_Cox():
    # 1.MinMax归一化操作
    # 2.进行Box_Cox转换
    # 对训练集与测试集合并处理即可！
    train = reduce_mem_usage(read_csv(tmppath + "date_train.csv")).drop(drop_features, axis=1)
    test = reduce_mem_usage(read_csv(tmppath + "date_test.csv")).drop(drop_features, axis=1)
    continue_list = list(set(train.columns.tolist()) - set(discrete_list) - {'emd_lable2'})
    from sklearn.preprocessing import MinMaxScaler
    data_all = pd.concat([train, test])
    minmax = MinMaxScaler()
    data_all[continue_list] = minmax.fit_transform(data_all[continue_list])
    # 对所有的连续型特征进行转换
    for column in continue_list:
        data_all[column], lmbda = boxcox(data_all[column] + 1)
    # 连续数据转换完成后保存文件,已经去除了大量特征
    train = data_all[0:23432].to_csv(tmppath + 'Box_train.csv', index=False)
    test = data_all[23432:].to_csv(tmppath + 'Box_test.csv', index=False)
    return train, test


def getTrainTest(X, Y):
    global x_train, x_test, y_train, y_test
    # 会员编号等，等下仔细去查看所有取值数量超过100的特征
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kfold.split(X, Y):
        x_train = X.loc[train_index]
        x_test = X.loc[test_index]
        y_train = Y.loc[train_index]
        y_test = Y.loc[test_index]
        break
    return x_train, x_test, y_train, y_test


def getTrainTest_np(X, Y):
    global x_train, x_test, y_train, y_test
    # 会员编号等，等下仔细去查看所有取值数量超过100的特征
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kfold.split(X, Y):
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = Y[train_index]
        y_test = Y[test_index]
        break
    return x_train, x_test, y_train, y_test


def minmax_target(X_train, X_test, Y_train, continue_list, discrete_list):
    import category_encoders as ce
    from sklearn.preprocessing import MinMaxScaler

    encoder = ce.LeaveOneOutEncoder(cols=discrete_list, drop_invariant=False).fit(X_train, Y_train)
    minmax = MinMaxScaler()
    train = pd.concat([X_train, X_test])
    minmax.fit(train[continue_list])

    X_train = encoder.transform(X_train)  # 基于训练集得到编码器
    X_test = encoder.transform(X_test)
    X_train[continue_list] = minmax.transform(X_train[continue_list])
    X_test[continue_list] = minmax.transform(X_test[continue_list])
    return X_train, X_test


def target(X_train, X_test, Y_train, discrete_list):
    import category_encoders as ce

    encoder = ce.LeaveOneOutEncoder(cols=discrete_list, drop_invariant=False).fit(X_train, Y_train)

    X_train = encoder.transform(X_train)  # 基于训练集得到编码器
    X_test = encoder.transform(X_test)
    return X_train, X_test


def minmax(X_train, X_test, continue_list):
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    train = pd.concat([X_train, X_test])
    minmax.fit(train[continue_list])
    X_train[continue_list] = minmax.transform(X_train[continue_list])
    X_test[continue_list] = minmax.transform(X_test[continue_list])
    return X_train, X_test


# 组合交叉特征，提升模型的线性组合能力（数值特征的变化），特征1+特征2这种类型
def auto_feature_make(train, test, continue_list):
    for col_i in continue_list:
        for col_j in continue_list:
            for func_name, func in func_dict.items():
                for data in [train, test]:
                    func_features = func(data[col_i], data[col_j])
                    col_func_features = '_'.join([col_i, func_name, col_j])
                    data[col_func_features] = func_features
                    print(col_func_features)
    return train, test


# 数值特征与连续特征的组合，这种方法用到中位数，均值等，需要对所有数据操作
def combine_feature(train, test, discrete_list, continue_list):
    data = pd.concat([train, test])
    for col_i in discrete_list:
        for col_j in continue_list:
            for func in func_list:
                col_func_features = '_'.join([col_j, func, col_i])
                data[col_func_features] = data[col_j].groupby(data[col_i]).transform(func)
                print(col_func_features)
    for col_i in discrete_list:
        for col_j in discrete_list:
            col_func_features = '_'.join([col_j, 'count', col_i])
            data[col_func_features] = data[col_j].groupby(data[col_i]).transform('count')
            print(col_func_features)
    return data[0:23432], data[23432:]


import xgboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB


def stacking_clf(clf, train_x, train_y, test_x, clf_name, kf, label_split=None, folds=5):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        if clf_name in ["rf", "ada", "gb", "et", "lr", "knn", "gnb"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict_proba(te_x)

            train[test_index] = pre[:, 0].reshape(-1, 1)
            test_pre[i, :] = clf.predict_proba(test_x)[:, 0].reshape(-1, 1)

            cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x)
            params = {'booster': 'gbtree',
                      'objective': 'multi:softprob',
                      'eval_metric': 'mlogloss',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      "num_class": 2
                      }

            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)
                train[test_index] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                'boosting_type': 'gbdt',
                # 'boosting_type': 'dart',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                "num_class": 2,
                'silent': True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre = model.predict(te_x, num_iteration=model.best_iteration)
                train[test_index] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


def rf_clf(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestClassifier(n_estimators=1200, max_depth=20, n_jobs=-1, random_state=2017,
                                          max_features="auto", verbose=1)
    rf_train, rf_test = stacking_clf(randomforest, x_train, y_train, x_valid, "rf", kf, label_split=label_split)
    return rf_train, rf_test, "rf"


def ada_clf(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_clf(adaboost, x_train, y_train, x_valid, "ada", kf, label_split=label_split)
    return ada_train, ada_test, "ada"


def gb_clf(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingClassifier(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,
                                      max_depth=5, verbose=1)
    gbdt_train, gbdt_test = stacking_clf(gbdt, x_train, y_train, x_valid, "gb", kf, label_split=label_split)
    return gbdt_train, gbdt_test, "gb"


def et_clf(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesClassifier(n_estimators=1200, max_depth=35, max_features="auto", n_jobs=-1, random_state=2017,
                                     verbose=1)
    et_train, et_test = stacking_clf(extratree, x_train, y_train, x_valid, "et", kf, label_split=label_split)
    return et_train, et_test, "et"


def xgb_clf(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_clf(xgboost, x_train, y_train, x_valid, "xgb", kf, label_split=label_split)
    return xgb_train, xgb_test, "xgb"


def lgb_clf(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_clf(lightgbm, x_train, y_train, x_valid, "lgb", kf, label_split=label_split)
    return xgb_train, xgb_test, "lgb"


def gnb_clf(x_train, y_train, x_valid, kf, label_split=None):
    gnb = GaussianNB()
    gnb_train, gnb_test = stacking_clf(gnb, x_train, y_train, x_valid, "gnb", kf, label_split=label_split)
    return gnb_train, gnb_test, "gnb"


def lr_clf(x_train, y_train, x_valid, kf, label_split=None):
    logisticregression = LogisticRegression(n_jobs=-1, random_state=2017, C=0.1, max_iter=200)
    lr_train, lr_test = stacking_clf(logisticregression, x_train, y_train, x_valid, "lr", kf, label_split=label_split)
    return lr_train, lr_test, "lr"


def knn_clf(x_train, y_train, x_valid, kf, label_split=None):
    kneighbors = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
    knn_train, knn_test = stacking_clf(kneighbors, x_train, y_train, x_valid, "lr", kf, label_split=label_split)
    return knn_train, knn_test, "knn"


def get_model_list():
    from lightgbm.sklearn import LGBMClassifier
    clf1 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.05,
                          n_estimators=3450,
                          max_depth=7,
                          num_leaves=65,
                          max_bin=440,
                          min_data_in_leaf=20,
                          feature_fraction=1.0,
                          bagging_fraction=0.7,
                          bagging_freq=0,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf2 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.05,
                          n_estimators=7125,
                          max_depth=9,
                          num_leaves=30,
                          max_bin=460,
                          min_data_in_leaf=10,
                          feature_fraction=0.9,
                          bagging_fraction=0.7,
                          bagging_freq=0,
                          lambda_l1=1e-05,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf3 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.025,
                          n_estimators=3250,
                          max_depth=9,
                          num_leaves=75,
                          max_bin=340,
                          min_data_in_leaf=10,
                          feature_fraction=1.0,
                          bagging_fraction=0.7,
                          bagging_freq=0,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf4 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.1,
                          n_estimators=1250,
                          max_depth=7,
                          num_leaves=80,
                          max_bin=480,
                          min_data_in_leaf=10,
                          feature_fraction=0.7,
                          bagging_fraction=0.7,
                          bagging_freq=8,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf5 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.1,
                          n_estimators=1875,
                          max_depth=9,
                          num_leaves=35,
                          max_bin=440,
                          min_data_in_leaf=10,
                          feature_fraction=0.8,
                          bagging_fraction=0.9,
                          bagging_freq=4,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf6 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.025,
                          n_estimators=3900,
                          max_depth=7,
                          num_leaves=35,
                          max_bin=480,
                          min_data_in_leaf=10,
                          feature_fraction=0.8,
                          bagging_fraction=0.9,
                          bagging_freq=2,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf7 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.1,
                          n_estimators=4375,
                          max_depth=7,
                          num_leaves=50,
                          max_bin=420,
                          min_data_in_leaf=10,
                          feature_fraction=0.8,
                          bagging_fraction=0.9,
                          bagging_freq=12,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf8 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.1,
                          n_estimators=1875,
                          max_depth=9,
                          num_leaves=70,
                          max_bin=460,
                          min_data_in_leaf=10,
                          feature_fraction=1.0,
                          bagging_fraction=0.7,
                          bagging_freq=0,
                          lambda_l1=0.0,
                          lambda_l2=1e-05,
                          min_split_gain=0.0)
    clf9 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                          learning_rate=0.05,
                          n_estimators=5750,
                          max_depth=7,
                          num_leaves=90,
                          max_bin=460,
                          min_data_in_leaf=10,
                          feature_fraction=0.7,
                          bagging_fraction=0.8,
                          bagging_freq=16,
                          lambda_l1=0.0,
                          lambda_l2=0.0,
                          min_split_gain=0.0)
    clf10 = LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='f1',
                           learning_rate=0.1,
                           n_estimators=2300,
                           max_depth=9,
                           num_leaves=30,
                           max_bin=420,
                           min_data_in_leaf=20,
                           feature_fraction=1.0,
                           bagging_fraction=0.7,
                           bagging_freq=0,
                           lambda_l1=0.0,
                           lambda_l2=0.0,
                           min_split_gain=0.0)
    return [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10]


def VoteClassify(clf_list, X_test):
    predict_martix = []
    for i in range(len(clf_list)):
        x_test = X_test[feature_subset_list[i]]
        predict_martix.append(clf_list[i].predict_proba(x_test)[:, 1])
    # 对预测的概率矩阵进行分析
    output = []
    for col in range(len(predict_martix[0])):
        sum = 0
        for row in range(len(predict_martix)):
            sum += predict_martix[row][col]
        sum /= len(predict_martix)
        output.append(sum)
    return output


def ensemble(X_train, Y_train, X_test):
    '''
    使用soft voting classifier 将所有模型预测样本为某一类别的概率的平均值作为标准，概率最高的
    对应的类型为最终的预测结果！增大概率值更大的模型（更有把握的模型）的影响力
    :param train:全部训练集，将其十折交叉验证获取10组训练集/测试集，并分别用于训练与验证模型
    :param test: 全部测试集，采用集成学习的方式对其进行预测，5个模型或以上认为其为付费旅客则预测为1，否则为0
    :return: 获得的集成学习模型ensemble_model,预测旅客列表的概率[0.1,0.4,0.6,....,0.9]
    '''
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics
    kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    clf_list = get_model_list()
    for k, (train_index, test_index) in enumerate(kfold.split(X_train, Y_train)):
        feature_subset = feature_subset_list[k]
        x_train = X_train.loc[train_index][feature_subset]
        x_test = X_train.loc[test_index][feature_subset]
        y_train = Y_train.loc[train_index]
        y_test = Y_train.loc[test_index]
        print('第{0}折进行交叉验证'.format(k + 1))
        clf_list[k].fit(x_train, y_train)
        train_predict = clf_list[k].predict(x_train)
        test_predict = clf_list[k].predict(x_test)
        print('The f1 of the train is:', metrics.f1_score(y_train, train_predict), 'The auc of the train is',
              metrics.roc_auc_score(y_train, clf_list[k].predict_proba(x_train)[:, 1]))
        print('The f1 of the test is:', metrics.f1_score(y_test, test_predict), 'The auc of the test is',
              metrics.roc_auc_score(y_test, clf_list[k].predict_proba(x_test)[:, 1]))
    # 对10个模型进行集成
    output_prob = VoteClassify(clf_list, X_test)
    output_prob_train = VoteClassify(clf_list, X_train)
    return output_prob, output_prob_train


train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))
test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))

X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
Y_train = train['emd_lable2'].astype(int)

discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
feature_list = X_train.columns.tolist()
continue_list = list(set(feature_list) - set(discrete_list))

# 数据归一化与编码
X_train, X_test = minmax_target(X_train, X_test, Y_train, continue_list, discrete_list)
result, validation = ensemble(X_train, Y_train, X_test)

validation = np.array([1 if i > 0.5 else 0 for i in validation])
print(metrics.f1_score(y_true=Y_train, y_pred=validation))

pd.DataFrame(data=result).to_csv('result_test1.0.csv')
