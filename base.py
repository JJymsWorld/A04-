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


class emsemble:
    '''
    使用soft voting classifier 将所有模型预测样本为某一类别的概率的平均值作为标准，概率最高的
    对应的类型为最终的预测结果！增大概率值更大的模型（更有把握的模型）的影响力
    :param train:全部训练集，将其十折交叉验证获取10组训练集/测试集，并分别用于训练与验证模型
    :param test: 全部测试集，采用集成学习的方式对其进行预测，5个模型或以上认为其为付费旅客则预测为1，否则为0
    :return: 获得的集成学习模型ensemble_model,预测旅客列表的概率[0.1,0.4,0.6,....,0.9]
    '''
    import numpy as np
    import pandas as pd
    clf_list = None
    feature_subset_list = None
    clf_weight_sum = [0.43243243243243246, 0.3773584905660377, 0.4932735426008969, 0.4220183486238532,
                      0.4573991031390135
        , 0.4380952380952381, 0.4104803493449782, 0.4748858447488585, 0.42016806722689076, 0.45267489711934156]
    clf_weight = None

    def __init__(self, feature_subset_list):
        self.clf_list = self.get_model_list()
        self.feature_subset_list = feature_subset_list
        self.clf_weight = [i / np.sum(self.clf_weight_sum) for i in self.clf_weight_sum]

    def get_model_list(self):
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

    def predict(self, X_test):
        predict_martix = []
        for i in range(len(self.clf_list)):
            x_test = X_test[self.feature_subset_list[i]]
            predict_martix.append(self.clf_list[i].predict_proba(x_test)[:, 1])
        # 对预测的概率矩阵进行分析
        output = []
        for col in range(len(predict_martix[0])):
            sum = 0
            for row in range(len(predict_martix)):
                sum += predict_martix[row][col] * self.clf_weight[row]
            output.append(sum)
        return output, predict_martix

    def fit(self, X_train, Y_train):
        from sklearn.model_selection import StratifiedKFold
        from sklearn import metrics
        kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        for k, (train_index, test_index) in enumerate(kfold.split(X_train, Y_train)):
            feature_subset = self.feature_subset_list[k]
            x_train = X_train.loc[train_index][feature_subset]
            x_test = X_train.loc[test_index][feature_subset]
            y_train = Y_train.loc[train_index]
            y_test = Y_train.loc[test_index]
            print('第{0}折进行交叉验证'.format(k + 1))
            self.clf_list[k].fit(x_train, y_train)
            train_predict = self.clf_list[k].predict(x_train)
            test_predict = self.clf_list[k].predict(x_test)
            print('The f1 of the train is:', metrics.f1_score(y_train, train_predict), 'The auc of the train is',
                  metrics.roc_auc_score(y_train, self.clf_list[k].predict_proba(x_train)[:, 1]))
            print('The f1 of the test is:', metrics.f1_score(y_test, test_predict), 'The auc of the test is',
                  metrics.roc_auc_score(y_test, self.clf_list[k].predict_proba(x_test)[:, 1]))


#
# train = reduce_mem_usage(read_csv(tmppath + "BOX_train.csv"))
# test = reduce_mem_usage(read_csv(tmppath + 'BOX_test.csv'))
#
# X_train = train.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)  # 去除部分取值过多的离散型特征
# X_test = test.drop(['emd_lable2'], axis=1).drop(train_drop_features, axis=1)
# Y_train = train['emd_lable2'].astype(int)
#
# discrete_list = list(set(discrete_list) - set(train_drop_features))  # 训练中需要剔除的特征都是离散型的特征
# feature_list = X_train.columns.tolist()
# continue_list = list(set(feature_list) - set(discrete_list))
#
# # 数据归一化与编码
# X_train, X_test = minmax_target(X_train, X_test, Y_train, continue_list, discrete_list)
# import joblib
# model = joblib.load(tmppath+'saved_model_3_19_1.0_weighted.pkl')

# model = emsemble(feature_subset_list)
# model.fit(X_train, Y_train)
# output, predict_martix = model.predict(X_test)
#
# # Counter(np.array(output) > 0.5)
# import joblib
#
# joblib.dump(model, tmppath + 'saved_model_3_19_1.0_weighted.pkl')
# pd.DataFrame(data=predict_martix).to_csv(tmppath + 'result_test1.0_weighted.csv')
allin = set()
for subset in feature_subset_list:
    allin = allin | set(subset)
    print(allin)
print("最终得到的所有子集为", allin)
pd.DataFrame(data=allin).to_csv(tmppath + 'feature_subset.csv')
pd.DataFrame(data=feature_subset_list).to_csv(tmppath + 'feature_subset_list.csv')

myself = ('seg_dep_time_month',
          'seg_dep_time_year',
          'seg_dep_time_is_workday',
          'seg_dep_time_is_holiday',
          'seg_dep_time_is_in_lieu',
          'recent_flt_day_month',
          'recent_flt_day_year',
          'recent_flt_day_hour',
          'recent_flt_day_is_workday',
          'recent_flt_day_is_holiday',
          'recent_flt_day_is_in_lieu',
          'birth_interval_day',
          'last_flt_day',
          'has_ffp_nbr',
          'seg_dep_time_hour')