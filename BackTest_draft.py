
import vectorbt as vbt
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.graph_objs as go
from vectorbt.indicators.factory import IndicatorFactory
import pytz  # Make sure pytz is installed

def custom_indicator(close, rsi, ma):
    close_5m = close.resample("5T").last()
    #rsi = vbt.RSI.run(close_5m, window = rsi_window).rsi
    rsi, _ = rsi.align(close, 
            broadcast_axis=0,
            method='ffill',
            join='right')

    close = close.to_numpy()
    rsi = rsi.to_numpy()
    #ma = vbt.MA.run(close, ma_window).ma.to_numpy()
    trend = np.where( rsi > 70, -1, 0)
    trend = np.where( (rsi < 30) & (close < ma), 1, trend)
    return trend

# def run_entry_exit(close, rsi, ma):
#     close_5m = close.resample("5T").last()
#     #rsi = vbt.RSI.run(close_5m, window = rsi_window).rsi

#     rsi, _ = rsi.align(close, 
#             broadcast_axis=0,
#             method='ffill',
#             join='right')
#     close = close.to_numpy()
#     rsi = rsi.to_numpy()
#     ma = ma.to_numpy()

#     trend = np.where( rsi > 70, -1, 0)

#     trend = np.where( (rsi < 30) & (close < ma), 1, trend)
#     #import pdb; pdb.set_trace()
#     return trend

def run_entry_exit(entries,exits):
    # close_5m = close.resample("5T").last()
    # #rsi = vbt.RSI.run(close_5m, window = rsi_window).rsi

    # rsi, _ = rsi.align(close, 
    #         broadcast_axis=0,
    #         method='ffill',
    #         join='right')
    # close = close.to_numpy()
    # rsi = rsi.to_numpy()
    # ma = ma.to_numpy()

    trend = np.where( exits, -1, 0)

    trend = np.where( entries, 1, trend)
    #import pdb; pdb.set_trace()
    return trend

def create_indicator(close,indicator1, ind1_dict):
    params = list(indicator1.param_names)
    params.remove("fillna")
    param_len = len(params)

    #for param, paramval in ind1_dict.items():

    #ind_final2 = indicator1.run(close, window = 15)
    ind_final = indicator1.run(close, **ind1_dict)
    
    #for outputname in ind_final.output_names:
    outputname = ind_final.output_names[0]
    #outputname2 = ind_final2.output_names[0]

    output_data = getattr(ind_final, outputname)
    return output_data
    

def run_backtest(close,trend):
    # btc_price = vbt.YFData.download(
    #         ["BTC-USD","ETH-USD"],
    #         missing_index='drop',
    #         start=start_time,
    #         end=end_time,
    #         interval="1m").get("Close")



    # ind = vbt.IndicatorFactory(
    #         class_name = "Combination",
    #         short_name = "comb",
    #         input_names = ["close"],
    #         param_names = ["rsi_window", "ma_window"],
    #         output_names = ["value"]
    #         ).from_apply_func(
    #                 custom_indicator,
    #                 rsi_window = 14,
    #                 ma_window = 50,
    #                 keep_pd=True
    #                 )

    # res = ind.run(
    #         close,
    #         rsi_window = 21,
    #         ma_window = 50
    #         )

    entries = trend == 1.0
    exits = trend == -1.0
    #import pdb; pdb.set_trace()
    pf = vbt.Portfolio.from_signals(close, entries, exits)
    #import pdb; pdb.set_trace()
    return pf

def compute_condition(indicator, operator, value):

    if operator == '>':
        return indicator > value
    elif operator == '<':
        return indicator < value
    elif operator == '==':
        return indicator == value
    elif operator == '>=':
        return indicator >= value
    elif operator == '<=':
        return indicator <= value
    elif operator == 'Crosses Above':
        return indicator.vbt.crossed_above(value)
    elif operator == 'Crosses Below':
        return indicator.vbt.crossed_below(value)
    else:
        st.error(f"Unsupported operator: {operator}")
        return None

def compute_multiple_conditions(selected_indicator1, ind1,selected_indicator2,ind2, close_price,conditions_list, condition_combine_logic):
        condition_bool_set = False
        for i in range(len(st.session_state["entry_condition"])):
            condition_i = st.session_state["entry_condition"][i]
            condition_ind_name_i = condition_i.get("ind_to_compare")
            entry_operator = condition_i.get("operator")
            entry_value = condition_i.get("value")
            
            #assing indicator value based on the condition indicator name
            if condition_ind_name_i == selected_indicator1:
                condition_ind_value_i = ind1
            elif condition_ind_name_i == selected_indicator2:
                condition_ind_value_i = ind2

            # check if entry value is string or float. in case of string get float values to compare
            if type(entry_value) == str:
                if entry_value == selected_indicator1:
                    entry_value = ind1
                elif entry_value == selected_indicator2:
                    entry_value = ind2
                elif entry_value == "close_price":
                    entry_value = close_price

            condition_i_bool = compute_condition(condition_ind_value_i, entry_operator, entry_value)

            if not condition_bool_set:
                condition_bool = condition_i_bool
                condition_bool_set = True
            else:
                
                if st.session_state["entry_condition_logic"] == "AND":
                    condition_bool = condition_bool & condition_i_bool
                elif st.session_state["entry_condition_logic"] == "OR":
                    condition_bool = condition_bool | condition_i_bool

        return condition_bool

def compute_multiple_conditions(selected_indicator1, ind1,selected_indicator2,ind2, close_price,conditions_list, condition_combine_logic):
        condition_bool_set = False
        for i in range(len(conditions_list)):
            condition_i = conditions_list[i]
            condition_ind_name_i = condition_i.get("ind_to_compare")
            entry_operator = condition_i.get("operator")
            entry_value = condition_i.get("value")
            
            #assing indicator value based on the condition indicator name
            if condition_ind_name_i == selected_indicator1:
                condition_ind_value_i = ind1
            elif condition_ind_name_i == selected_indicator2:
                condition_ind_value_i = ind2

            # check if entry value is string or float. in case of string get float values to compare
            if type(entry_value) == str:
                if entry_value == selected_indicator1:
                    entry_value = ind1
                elif entry_value == selected_indicator2:
                    entry_value = ind2
                elif entry_value == "close_price":
                    entry_value = close_price

            condition_i_bool = compute_condition(condition_ind_value_i, entry_operator, entry_value)

            if not condition_bool_set:
                condition_bool = condition_i_bool
                condition_bool_set = True
            else:
                
                if condition_combine_logic == "AND":
                    condition_bool = condition_bool & condition_i_bool
                elif condition_combine_logic == "OR":
                    condition_bool = condition_bool | condition_i_bool

        return condition_bool


def define_indicators_category():
    indicator_categories = {
        'TRIXIndicator': 'Momentum',
        'PercentagePriceOscillator': 'Momentum',
        'ChaikinMoneyFlowIndicator': 'Volume',
        'NegativeVolumeIndexIndicator': 'Volume',
        'OnBalanceVolumeIndicator': 'Volume',
        'PSARIndicator': 'Trend',
        'DailyLogReturnIndicator': 'Returns',
        'EMAIndicator': 'Trend',
        'WMAIndicator': 'Trend',
        'WilliamsRIndicator': 'Momentum',
        'MFIIndicator': 'Volume',
        'RSIIndicator': 'Momentum',
        'IchimokuIndicator': 'Trend',
        'KeltnerChannel': 'Volatility',
        'SMAIndicator': 'Trend',
        'ADXIndicator': 'Trend',
        'KSTIndicator': 'Momentum',
        'StochRSIIndicator': 'Momentum',
        'AccDistIndexIndicator': 'Volume',
        'ForceIndexIndicator': 'Volume',
        'StochasticOscillator': 'Momentum',
        'MassIndex': 'Trend',
        'EaseOfMovementIndicator': 'Volume',
        'MACD': 'Momentum',
        'DPOIndicator': 'Momentum',
        'BollingerBands': 'Volatility',
        'UltimateOscillator': 'Momentum',
        'ROCIndicator': 'Momentum',
        'AroonIndicator': 'Trend',
        'STCIndicator': 'Trend',
        'AverageTrueRange': 'Volatility',
        'VolumePriceTrendIndicator': 'Volume',
        'KAMAIndicator': 'Trend',
        'DonchianChannel': 'Volatility',
        'UlcerIndex': 'Volatility',
        'VortexIndicator': 'Trend',
        'AwesomeOscillatorIndicator': 'Momentum',
        'VolumeWeightedAveragePrice': 'Volume',
        'CumulativeReturnIndicator': 'Returns',
        'DailyReturnIndicator': 'Returns',
        'PercentageVolumeOscillator': 'Volume',
        'CCIIndicator': 'Momentum',
        'TSIIndicator': 'Momentum'
        }
    return indicator_categories

def toggle():
    if st.session_state.button:
        st.session_state.button = False
    else:
        st.session_state.button = True

def refresh_entry_container():
    st.session_state["refresh_entrycontainer_count"] += 1
    #st.session_state["entry_ind_options"].pop()
 
    st.session_state["selected_entry_ind_to_compare"] = None
    st.session_state["selected_operator_compare"] = None
    st.session_state["selected_value_compare"] = None

def refresh_exit_container():
    st.session_state["refresh_exitcontainer_count"] += 1
    #st.session_state["entry_ind_options"].pop()
 
    st.session_state["selected_exit_ind_to_compare"] = None
    st.session_state["selected_exit_operator_compare"] = None
    st.session_state["selected_exit_value_compare"] = None

def toggle_condition_container(entry_ind_to_compare,entry_operator,entry_value,condition_logic,condition_key="entry_condition"):
    add_condition(entry_ind_to_compare,entry_operator,entry_value,condition_key)
    if condition_key == "entry_condition":
        refresh_entry_container()
    
        st.session_state["entry_condition_logic"] = condition_logic

        if st.session_state.toggle_entry_container:
            st.session_state.toggle_entry_container = False
        else:
            st.session_state.toggle_entry_container = True
    elif condition_key == "exit_condition":
        refresh_exit_container()
    
        st.session_state["exit_condition_logic"] = condition_logic

        if st.session_state.toggle_exit_container:
            st.session_state.toggle_exit_container = False
        else:
            st.session_state.toggle_exit_container = True


def add_condition(entry_ind_to_compare,entry_operator,entry_value,condition_key):

    total_conditions = len(st.session_state[condition_key]) + 1

    st.session_state[condition_key].append({
        "ind_to_compare": entry_ind_to_compare,       # Field/parameter to check
        "operator": entry_operator,    # Condition operator (e.g., >, <, ==)
        "value": entry_value,        # Value to compare
        "s_no_entry_condition" : total_conditions 
        })



# Convert date to datetime with timezone
def convert_to_timezone_aware(date_obj):
    return datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=pytz.UTC)

def main():
    if "button" not in st.session_state:
        st.session_state.button = False

    if "toggle_entry_container" not in st.session_state:
        st.session_state.toggle_entry_container = False

    if "toggle_exit_container" not in st.session_state:
        st.session_state.toggle_exit_container = False

    # Initialize session state for conditions
    if "entry_condition" not in st.session_state:
        st.session_state["entry_condition"] = []  # List of conditions

    if "exit_condition" not in st.session_state:
        st.session_state["exit_condition"] = []  # List of conditions

    
    if "refresh_entrycontainer_count" not in st.session_state:
        st.session_state["refresh_entrycontainer_count"] = 0

    if "refresh_exitcontainer_count" not in st.session_state:
        st.session_state["refresh_exitcontainer_count"] = 0

    if "selected_entry_ind_to_compare" not in st.session_state:
        st.session_state["selected_entry_ind_to_compare"] = None
    
    if "selected_exit_ind_to_compare" not in st.session_state:
        st.session_state["selected_exit_ind_to_compare"] = None

    if "selected_value_compare" not in st.session_state:
        st.session_state["selected_value_compare"] = None

    if "selected_exit_value_compare" not in st.session_state:
        st.session_state["selected_exit_value_compare"] = None
        
    if "selected_operator_compare" not in st.session_state:
        st.session_state["selected_operator_compare"] = None

    if "selected_exit_operator_compare" not in st.session_state:
        st.session_state["selected_exit_operator_compare"] = None


    if "comparison_operators" not in st.session_state:
        st.session_state["comparison_operators"] = ['>', '<', '==', '>=', '<=', 'Crosses Above', 'Crosses Below']

    if "entry_ind_options" not in st.session_state:
        st.session_state["entry_ind_options"] = []

    if "comparison_value" not in st.session_state:
        st.session_state["comparison_value"] = ['Second Indicator', 'A Number', 'Close Price', 'Other Indicator']

    if "entry_condition_logic" not in st.session_state:
        st.session_state["entry_condition_logic"] = None

    if "exit_condition_logic" not in st.session_state:
        st.session_state["exit_condition_logic"] = None

    indicator_definitions = define_indicators_category()

    instruments_name_list = ['Gold','NASDAQ-100','GBPJPY','GBPUSD','EURUSD']
    instruments_symbols= {'Gold':'GC=F','NASDAQ-100':'^NDX','GBPJPY':'GBPJPY=X', 'GBPUSD':'GBPUSD=X', 'EURUSD':'EURUSD=X'}

    interval_mapping = {'5 minute': '5m',
                        '15 minute': '15m',
                        '1 hour': '1h',
                        '1 day': '1d', 
                        '1 week': '1wk', 
                        '1 month': '1mo' }
    
    ohlc = ['Open', 'High', 'Low', 'Close', #'Volume'
            ]

    # all indicators
    #strategy_names = list(vbt.IndicatorFactory.get_ta_indicators())
    #selected indicators
    strategy_names = ['RSIIndicator','SMAIndicator','EMAIndicator','BollingerBands','IchimokuIndicator','StochasticOscillator','VolumeWeightedAveragePrice']


    st.title('Backtest Trading Strategies')
    # Sidebar for inputs
    with st.sidebar:
        # Inputs for the symbol, start and end dates
        st.header("Strategy Controls")
        
        # Inputs for the symbol, start and end dates
        instrument_name = st.selectbox("Enter the symbol (e.g., 'AAPL')",instruments_name_list  )
        start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
        time_period = st.selectbox('Interval', ['5 minute','15 minute','1 hour','1 day','1 week','1 month'], index = 2)
        data_ohlc = st.selectbox('OHLC', ohlc , index = 3)

        st.header("Indicator Controls")

        selected_indicator1 = st.selectbox('Select Indicator 1', strategy_names,index = None)
        if selected_indicator1 is not None:
            indicator1 = vbt.IndicatorFactory.from_ta( selected_indicator1,init_kwargs=None,)
            ind1_dict ={}
            for i in list(indicator1.param_names):
                if i != "fillna":
                    ind1_dict[i] =st.number_input(i, value=15,key = f'1_{selected_indicator1}_{i}')
        
        selected_indicator2 = st.selectbox('Select Indicator 2', strategy_names,index = None)
        if selected_indicator2 is not None:
            indicator2 = vbt.IndicatorFactory.from_ta( selected_indicator2,init_kwargs=None,)
        #ind2 = indicator2
            ind2_dict ={}
            for i in list(indicator2.param_names):
                if i != "fillna":
                    ind2_dict[i] =st.number_input(i, value=15, key = f'2_{selected_indicator2}_{i}')


    #symbol = st.text_input('Enter symbol (e.g., BTC-USD, AMZN, ...):', 'BTC-USD')
    #start_date = st.date_input('Select start date:', pd.to_datetime('2023-01-01'))
    #end_date = st.date_input('Select end date:', pd.to_datetime('2023-12-31'))

    
        st.button('Define Entry and Exit Conditions', on_click=toggle)
        #if st.button('Define Entry and Exit Conditions',on_click=toggle):

    if st.session_state.button == True:
        st.header("Define Entry and Exit Conditions")
        st.session_state["entry_ind_options"] = [selected_indicator1,selected_indicator2]
        entry_container = st.empty()
        with entry_container.container(border = True):
                st.subheader("Entry Condition")

                if len(st.session_state["entry_condition"]) > 0 :
                        entry_df = pd.DataFrame(st.session_state["entry_condition"]).reset_index(drop=True).reset_index(names="SN")  
                        st.dataframe(entry_df, use_container_width =True, column_order=("SN", "ind_to_compare","operator","value") , hide_index =True, key="entry_condition_df")

                with st.expander("Submit Entry Condition",expanded=st.session_state.button):
                    
                
                    st.write("Condition #:", len(st.session_state["entry_condition"]) + 1 )
                    #len(st.session_state[condition_key]) + 1
                    
                    entry_ind_to_compare = st.selectbox("Select indicator to compare:",st.session_state["entry_ind_options"],
                    index=None if st.session_state["selected_entry_ind_to_compare"] is None else st.session_state["entry_ind_options"].index(st.session_state["selected_entry_ind_to_compare"])
                    , key=f"entry_ind_to_compare_{st.session_state['refresh_entrycontainer_count']}")
                    st.session_state["selected_entry_ind_to_compare"] = entry_ind_to_compare


                    # Comparison Operator for Entry Condition
                    entry_operator = st.selectbox(  
                        'Select Comparison Operator',
                        st.session_state["comparison_operators"],
                        index=None if st.session_state["selected_operator_compare"] is None else st.session_state["comparison_operators"].index(st.session_state["selected_operator_compare"]),
                        key=f"entry_operator_{st.session_state['refresh_entrycontainer_count']}"
                    )
                    st.session_state["selected_operator_compare"] = entry_operator

                    # Comparison Type for Entry Condition

                    entry_comp_type = st.selectbox(
                        'Compare the indicator with:',
                        st.session_state["comparison_value"],
                        index=None if st.session_state["selected_value_compare"] is None else st.session_state["comparison_value"].index(st.session_state["selected_value_compare"]),
                        key=f"entry_comp_type_{st.session_state['refresh_entrycontainer_count']}"
                    )
                    st.session_state["selected_value_compare"] = entry_comp_type

                    entry_value = None
                    if entry_comp_type == 'Second Indicator':
                        if entry_ind_to_compare == selected_indicator1:
                            entry_value = selected_indicator2
                        elif entry_ind_to_compare == selected_indicator2:
                            entry_value = selected_indicator1
                    elif entry_comp_type == 'A Number':
                        entry_number = st.number_input('Enter the number to compare with', key='entry_number_entry')
                        entry_value = entry_number
                    elif entry_comp_type == 'Close Price':
                        entry_value = "close_price"
                    elif entry_comp_type == 'Other Indicator':
                        # Allow user to select another indicator and parameters
                        entry_strategy_names = [key for key, value in indicator_definitions.items()  if value == indicator_definitions.get(entry_ind_to_compare)]
                        other_indicator_name = st.selectbox('Select the other indicator', entry_strategy_names, key='entry_other_indicator_container')
                        other_indicator_en = vbt.IndicatorFactory.from_ta( other_indicator_name,init_kwargs=None,)
                        oth_ind_dict_en ={}
                        for i in list(other_indicator_en.param_names):
                            if i != "fillna":
                                oth_ind_dict_en[i] =st.number_input(i, value=15,key = f'en_{other_indicator_name}_{i}')
                        entry_value = create_indicator(close_price, other_indicator_en, oth_ind_dict_en)
                    

                    if len(st.session_state["entry_condition"]) > 0 :
                        if  st.session_state["entry_condition_logic"] is None:
                            entry_condition_logic_radio_disabled = False
                        else: 
                            entry_condition_logic_radio_disabled = True

                        entry_condition_logic = st.radio("Combine Conditions Using:", options=["AND", "OR"], index= 0 if st.session_state["entry_condition_logic"] is None else ["AND", "OR"].index(st.session_state["entry_condition_logic"]),  key="entry_condition_logic_radio", disabled=entry_condition_logic_radio_disabled)
                    else: 
                        entry_condition_logic = None

                if st.button("Save/Submit Condition",key="entryconditionbutton_container", 
                  on_click=toggle_condition_container, args = [entry_ind_to_compare,entry_operator,entry_value,entry_condition_logic,"entry_condition"]):
                        
                        st.write("Entry Condition ",len(st.session_state["entry_condition"]), " Submitted")
                                    # st.session_state["show_logic_popup"] = True  # Show AND/OR logic popup
                                        
        exit_container = st.empty()
        with exit_container.container(border = True):
            st.subheader("Exit Condition")
            if len(st.session_state["exit_condition"]) > 0 :
                        exit_df = pd.DataFrame(st.session_state["exit_condition"]).reset_index(drop=True).reset_index(names="SN")  
                        st.dataframe(exit_df, use_container_width =True, column_order=("SN", "ind_to_compare","operator","value") , hide_index =True, key="exit_condition_df")

            with st.expander("Submit Exit Condition",expanded=st.session_state.button):
                st.write("Condition #:", len(st.session_state["exit_condition"]) + 1 )

                exit_ind_to_compare = st.selectbox("Select indicator to compare:",st.session_state["entry_ind_options"],
                    index=None if st.session_state["selected_exit_ind_to_compare"] is None else st.session_state["entry_ind_options"].index(st.session_state["selected_exit_ind_to_compare"])
                    , key=f"exit_ind_to_compare_{st.session_state['refresh_exitcontainer_count']}")
                st.session_state["selected_exit_ind_to_compare"] = entry_ind_to_compare

                # Comparison Operator for Exit Condition
                exit_operator = st.selectbox(  
                        'Select Comparison Operator',
                        st.session_state["comparison_operators"],
                        index=None if st.session_state["selected_exit_operator_compare"] is None else st.session_state["comparison_operators"].index(st.session_state["selected_exit_operator_compare"]),
                        key=f"exit_operator_{st.session_state['refresh_exitcontainer_count']}"
                    )
                st.session_state["selected_exit_operator_compare"] = exit_operator

                # Comparison Type for Exit Condition

                exit_comp_type = st.selectbox(
                        'Compare the indicator with:',
                        st.session_state["comparison_value"],
                        index=None if st.session_state["selected_exit_value_compare"] is None else st.session_state["comparison_value"].index(st.session_state["selected_exit_value_compare"]),
                        key=f"exit_comp_type_{st.session_state['refresh_exitcontainer_count']}"
                    )
                st.session_state["selected_exit_value_compare"] = exit_comp_type

                # Exit Comparison Value Input
                exit_value = None
                if exit_comp_type == 'Second Indicator':
                    if exit_ind_to_compare == selected_indicator1:
                        exit_value = selected_indicator2
                    elif exit_ind_to_compare == selected_indicator2:
                        exit_value = selected_indicator1
                elif exit_comp_type == 'A Number':
                    exit_number = st.number_input('Enter the number to compare with', key='exit_number_exit')
                    exit_value = exit_number
                elif exit_comp_type == 'Close Price':
                    exit_value = "close_price"
                elif exit_comp_type == 'Other Indicator':
                    # Allow user to select another indicator and parameters
                    exit_strategy_names = [key for key, value in indicator_definitions.items()  if value == indicator_definitions.get(exit_ind_to_compare)]
                    other_indicator_name_exit = st.selectbox('Select the other indicator', exit_strategy_names, key='exit_other_indicator')
                    other_indicator_exit = vbt.IndicatorFactory.from_ta( other_indicator_name_exit,init_kwargs=None,)
                    oth_ind_dict_ex ={}
                    for i in list(other_indicator_exit.param_names):
                        if i != "fillna":
                            oth_ind_dict_ex[i] =st.number_input(i, value=15,key = f'ec_{other_indicator_name_exit}_{i}')
                    exit_value = create_indicator(close_price, other_indicator_exit, oth_ind_dict_ex)
                    # Input parameters for the other indicator
                    # Compute the other indicator
                    # other_window = st.number_input('Window Size for Other Indicator', min_value=1, value=20, key='exit_other_window')
                    # exit_value = vbt.MA.run(close_price, window=other_window).ma
                
                if len(st.session_state["exit_condition"]) > 0 :
                        if  st.session_state["exit_condition_logic"] is None:
                            exit_condition_logic_radio_disabled = False
                        else: 
                            exit_condition_logic_radio_disabled = True

                        exit_condition_logic = st.radio("Combine Conditions Using:", options=["AND", "OR"], index= 0 if st.session_state["exit_condition_logic"] is None else ["AND", "OR"].index(st.session_state["exit_condition_logic"]),  key="exit_condition_logic_radio", disabled=exit_condition_logic_radio_disabled)
                else: 
                        exit_condition_logic = None

            if st.button("Save/Submit Condition",key="exitcondition_button_container", 
                  on_click=toggle_condition_container, args = [exit_ind_to_compare,exit_operator,exit_value,exit_condition_logic,"exit_condition"]):
                        
                        st.write("Exit Condition ",len(st.session_state["exit_condition"])," Submitted")

               

    # condition_options = [
    #     'Crosses Above',
    #     'Crosses Below',
    #     'Greater Than',
    #     'Less Than',
    #     'Equal To'
    # ]

    # entry_condition = st.selectbox('Select Entry Condition', condition_options, key='entry_condition')
    # exit_condition = st.selectbox('Select Exit Condition', condition_options, key='exit_condition')


        if st.button('Run Backtest'):
            if  len(st.session_state["entry_condition"]) > 0 and  len(st.session_state["exit_condition"]) > 0:
                #end_time = datetime.datetime.now()
                #start_time = end_time - datetime.timedelta(days=2)
                start_date_tz = convert_to_timezone_aware(start_date)
                end_date_tz = convert_to_timezone_aware(end_date)
                symbol = instruments_symbols.get(instrument_name)
                interval =  interval_mapping.get(time_period)
                close_price = vbt.YFData.download(
                        [symbol],
                        missing_index='drop',
                        start=start_date_tz,
                        end=end_date_tz,
                        interval=interval).get(data_ohlc)
                
                ind1 = create_indicator(close_price, indicator1, ind1_dict)
                ind2 = create_indicator(close_price, indicator2, ind2_dict)


                # #create other indicator to compare; if selected for entry condition
                # if entry_comp_type ==  'Other Indicator':
                #     oth_ind_en = create_indicator(close_price, other_indicator_en, oth_ind_dict_en)
                #     entry_value = oth_ind_en

                #create other indicator to compare; if selected for exit condition
                if entry_comp_type ==  'Other Indicator':
                    oth_ind_en = create_indicator(close_price, other_indicator_exit, oth_ind_dict_ex)
                    exit_value = oth_ind_en

                entries = compute_multiple_conditions(selected_indicator1, ind1,selected_indicator2,ind2, close_price, st.session_state["entry_condition"],st.session_state["entry_condition_logic"] )
                #st.session_state["entry_condition_logic"]
                #entries = compute_condition(ind1, entry_operator, entry_value)
                if entries is None:
                    st.stop()
                # Ensure entries are boolean
                entries = entries.astype(bool)

                # Compute Exit Signals
                #exits = compute_condition(ind1, exit_operator, exit_value)
                exits = compute_multiple_conditions(selected_indicator1, ind1,selected_indicator2,ind2, close_price, st.session_state["exit_condition"],st.session_state["exit_condition_logic"] )
                if exits is None:
                    st.stop()
                # Ensure exits are boolean
                exits = exits.astype(bool)

                trend = run_entry_exit(entries,exits)
                pf = run_backtest(close_price,trend)
                
                # Create tabs
                print("creating tabs")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtesting Stats", "List of Trades", 
                                                    "Equity Curve", "Drawdown", "Portfolio Plot"])



                with tab1:
                    # Display results
                    st.markdown("**Backtesting Stats:**")

                    stats_df = pf.stats().to_frame().rename(columns={'agg_func_mean':'Value'})
                    stats_df.index.name = 'Metric'  # Set the index name to 'Metric' to serve as the header
                    st.dataframe(stats_df, height=800)  # Adjust the height as needed to remove the scrollbar

                with tab2:
                    st.markdown("**List of Trades:**")
                    trades_df = pf.trades.records_readable
                    trades_df = trades_df.round(2)  # Rounding the values for better readability
                    trades_df.index.name = 'Trade No'  # Set the index name to 'Trade Name' to serve as the header
                    trades_df.drop(trades_df.columns[[0,1]], axis=1, inplace=True)
                    st.dataframe(trades_df, width=800,height=600)  # Set index to False and use full width

                # Plotting
                equity_data = pf.value()
                drawdown_data = pf.drawdown() * 100

                with tab3:
                # Equity Curve
                    equity_trace = go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name='Equity',line=dict(color='green') )
                    equity_fig = go.Figure(data=[equity_trace])
                    equity_fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity',
                                                width=800,height=600)
                    st.plotly_chart(equity_fig)

                with tab4:
                    # Drawdown Curve
                    drawdown_trace = go.Scatter(
                        x=drawdown_data.index,
                        y=drawdown_data,
                        mode='lines',
                        name='Drawdown',
                        fill='tozeroy',
                        line=dict(color='red')  # Set the line color to red
                    )
                    drawdown_fig = go.Figure(data=[drawdown_trace])
                    drawdown_fig.update_layout(
                        title='Drawdown Curve',
                        xaxis_title='Date',
                        yaxis_title='% Drawdown',
                        template='plotly_white',
                        width = 800,
                        height = 600
                    )
                    st.plotly_chart(drawdown_fig)

                with tab5:
                    st.write(pf.stats())
                #     # Portfolio Plot
                #     st.markdown("**Portfolio Plot:**")
                #     st.plotly_chart(pf.plot())
            
            else:
                if len(st.session_state["entry_condition"]) > 0:
                    st.write("Exit Condition not submitted. Submit atleast one exit condition.")
                else:
                    st.write("Entry Condition not submitted. Submit atleast one exit condition.")


if __name__ == '__main__':
    main()