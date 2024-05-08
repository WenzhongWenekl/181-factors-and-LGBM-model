#%%
from posixpath import dirname
from typing_extensions import final
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from functions import *
from factors import *
import copy
import os
# from scipy.stats.mstats import winsorize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import bartlett
import scipy.stats as st
from factor_analyzer import *
import numpy.linalg as nlg
from matplotlib import cm
import math
import matplotlib.pyplot as plt  
import statsmodels.api as sm
plt.style.use('default')   
import warnings
warnings.filterwarnings("ignore")
pd.set_option('precision',4)
dirname = '/home/candidate/cand2/output'
if os.path.exists(dirname):
    pass
else:
    os.makedirs(dirname)
#%%
class testFactors:
    def __init__(self, filePath = '/home/candidate/cand3/factor_evaluation/data/stock_hfq/', startdate = '2010-01-01', enddate = '2018-12-31', num_alpha = 5, num_stock_file = 20):
        self.filePath = filePath
        self.startdate = startdate
        self.enddate = enddate
        self.num_alpha = num_alpha
        self.num_stock_flie = num_stock_file
        year_time = pd.to_datetime(startdate).strftime('%Y')
        month_day = pd.to_datetime(startdate).strftime('%m-%d')
        self.startdate_1 = '{}-{}'.format(str(int(year_time) - 1), month_day)
        # pass

        def __iter__(self):
                return self

    def Return(self, df, n):
    # this to get the 'return' used in IC & IR
        VWAP_1 = df.shift(-1)
        VWAP_n_1 = df.shift(-n-1)
        return VWAP_n_1 / VWAP_1

    def name(self):
        # Read in the names of all the files in the special directory
        file_name = []
        list_data = os.listdir(self.filePath)
        for j in list_data:
        # Determine if it is a 'txt' file, if so, store it in the list
            if os.path.splitext(j)[1] == '.csv':
                file_name.append(j)
        return file_name

    def test_alpha(self, filename):
    # Calculate all alpha factors in a stock
        # self.file_name = self.name()[:5]
        # factors  =  pd.DataFrame()
        # for filename in self.file_name:
        try:
            Data = pd.read_csv(self.filePath + filename) # Get date data
            Data['TRADE_DT']=pd.to_datetime(Data['TRADE_DT'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
            Data = Data[(Data['TRADE_DT'] >= self.startdate_1) & (Data['TRADE_DT'] <= self.enddate)]
            Data = Data.reset_index()
            print('Running...',filename)
            # Factor_i = Data[['S_DQ_ADJCLOSE', 'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_VOLUME', 'S_DQ_AMOUNT']]
            # Factor_i.columns = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUMN', 'AMOUNT']
            # Factor_i['VWAP'] = Factor_i['AMOUNT'] / Factor_i['VOLUMN']
            # Factor_i['stock_id'] = filename[:-4]
            CLOSE = Data['S_DQ_ADJCLOSE']
            OPEN = Data['S_DQ_ADJOPEN']
            HIGH = Data['S_DQ_ADJHIGH']
            LOW = Data['S_DQ_ADJLOW']
            VOLUME = Data['S_DQ_VOLUME']
            AMOUNT = Data['S_DQ_AMOUNT']
            VWAP = AMOUNT / VOLUME
            RET = CLOSE.pct_change(periods=1)
            BANCHMARK_INDEX_CLOSE = BANCHMARK_INDEX(CLOSE)
            BANCHMARK_INDEX_OPEN = BANCHMARK_INDEX(OPEN)
            # Factor_i = pd.DataFrame({'name' :filename[:-4], 'trade_date':Data['日期'], 'alpha1':a1, 'alpha2' :a2, 'VWAP' :VWAP, 'return' :Return(VWAP, n)})
            Factor_i = pd.DataFrame({'stock_id' :filename[:-4], 'trade_date':Data['TRADE_DT'], 'VWAP' :VWAP, 'CLOSE' :CLOSE})
            # Factor_i['trade_date']=pd.to_datetime(Factor_i['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
            for i in range(1, self.num_alpha + 1):
                exec('a' + str(i)  + '= alpha' + str(i) + '(HIGH,LOW,OPEN,CLOSE,AMOUNT,RET,BANCHMARK_INDEX_CLOSE,VOLUME,VWAP,BANCHMARK_INDEX_OPEN)')
                exec('Factor_i[' + str(i) + '] = a' + str(i)) 
            # factors = factors.append(Factor_i)
            print(filename, 'is done!')
            return Factor_i
        except:
            pass

    def merge_data(self):
        if self.num_stock_flie == -1:
            name = self.name()
        else:
            name = self.name()[:self.num_stock_flie]
        executor = ProcessPoolExecutor(36)
        futures = executor.map(self.test_alpha, name)
        # print('Start joining...')
        factors = pd.concat(futures)
        factors = factors.reset_index()
        # factors['trade_date']=pd.to_datetime(factors['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
        self.factors = factors
        print("FACTORS DONE!!!!!!!!!!!!!!!!")
        # return self.factors

    def get_factors(self):
    # n = 1,3,5,10,20
        # self.merge_data()
        # self.factors = pd.read_csv('/home/candidate/cand3/LightGBM_trial/data/Result.csv', index_col=[0])  # 用已经读好的数据
        print("FACTORS DONE!!!!!!!!!!!!!!!!")
        self.factors1 = copy.deepcopy(self.factors)
        self.factors1.reset_index(drop=True,inplace=True)
        self.factors1['return'] = self.factors1.groupby('stock_id')['VWAP'].apply(lambda x: x.shift(-1) / x.shift(-2))
        self.factors3 = copy.deepcopy(self.factors)
        self.factors3.reset_index(drop=True,inplace=True)
        self.factors3['return'] = self.factors3.groupby('stock_id')['VWAP'].apply(lambda x: x.shift(-1) / x.shift(-4))
        self.factors5 = copy.deepcopy(self.factors)
        self.factors5.reset_index(drop=True,inplace=True)
        self.factors5['return'] = self.factors5.groupby('stock_id')['VWAP'].apply(lambda x: x.shift(-1) / x.shift(-6))
        self.factors10 = copy.deepcopy(self.factors)
        self.factors10.reset_index(drop=True,inplace=True)
        self.factors10['return'] = self.factors10.groupby('stock_id')['VWAP'].apply(lambda x: x.shift(-1) / x.shift(-11))
        self.factors20 = copy.deepcopy(self.factors)
        self.factors20.reset_index(drop=True,inplace=True)
        self.factors20['return'] = self.factors20.groupby('stock_id')['VWAP'].apply(lambda x: x.shift(-1) / x.shift(-21))
        index = (self.factors1['trade_date'] >= self.startdate) & (self.factors1['trade_date'] <= self.enddate)
        self.Sure_data1 =self.factors1[index].dropna()
        self.Sure_data3 =self.factors3[index].dropna()
        self.Sure_data5 =self.factors5[index].dropna()
        self.Sure_data10 =self.factors10[index].dropna()
        self.Sure_data20 =self.factors20[index].dropna()
        self.date_list = np.sort(self.Sure_data1['trade_date'].unique())

    def winsorize(self, df):
        base = df.median()
        median = base
        # print(base)
        n = len(df)
        df.where(df > base + n * abs(df - median), base + n * abs(df - median))
        df.where(df < base - n * abs(df - median), base - n * abs(df - median))
        return df

    def Stand_win(self, array):
        try:
            return preprocessing.scale(self.winsorize(array))
        except:
            pass

    def getIC(self, date_array, sure_data, i):
        empty = pd.DataFrame(columns=["IC", "trade_date"])
        for date in date_array:
            oneday_df = sure_data.loc[sure_data['trade_date']==date]
    # if we do not need to standardize and winsorize the data
            # signal_df = preprocessing.scale(winsorize(oneday_df, limits=[0.025, 0.025]))
    # if we try to standardize and winsorize the data
            pearson1 = pd.DataFrame({'return': self.Stand_win(oneday_df['return'].replace([np.inf,-np.inf],[10000,-10000])), 'factor': self.Stand_win(oneday_df[i].replace([np.inf,-np.inf],[10000,-10000]))})
            empty = empty.append([{'IC':pearson1.corr(method='pearson')['return']['factor'], 'trade_date':date}],ignore_index=True)
        empty['year']=pd.to_datetime(empty['trade_date'],format='%Y-%m-%d').dt.strftime('%Y')
        return empty

    def get_IC_Accmulative_IR_data(self):
        # self.date_list = self.date_list[(pd.to_datetime(self.date_list ,format = '%Y%m%d') >= pd.to_datetime(self.startdate,format = '%Y%m%d')) & (pd.to_datetime(self.date_list ,format = '%Y%m%d') <= pd.to_datetime('20181231',format = '%Y%m%d'))]
        for i in range(1, self.num_alpha + 1):
            dirname = '/home/candidate/cand2/output/Alpha' + str(i)
            # os.mkdir(dirname)
            if os.path.exists(dirname):
                pass
            else:
                os.makedirs(dirname)
            IC_Accumulative1 = SUMAC(self.getIC(self.date_list, self.Sure_data1,i)['IC'])
            # print(IC_Accumulative1)
            IC_Accumulative3 = SUMAC(self.getIC(self.date_list, self.Sure_data3,i)['IC'])
            IC_Accumulative5 = SUMAC(self.getIC(self.date_list, self.Sure_data5,i)['IC'])
            IC_Accumulative10 = SUMAC(self.getIC(self.date_list, self.Sure_data10,i)['IC'])
            IC_Accumulative20 = SUMAC(self.getIC(self.date_list, self.Sure_data20,i)['IC'])
            # IC_daily =  pd.DataFrame({'ic1': IC_list1,'ic3': IC_list3, 'ic5': IC_list5,'ic10': IC_list10,'ic20': IC_list20})
            IC_Accumulative =  pd.DataFrame({'ic1': IC_Accumulative1,'ic3': IC_Accumulative3, 'ic5': IC_Accumulative5,'ic10': IC_Accumulative10,'ic20': IC_Accumulative20})
            print("IC_Accumulative_daily is :", IC_Accumulative)
            IC_Accumulative.plot()
            plt.title('IC from 2010 to 2018 (Daily)' + 'Alpha' + str(i), color = '#6D6D6D', fontsize = 18)
            plt.legend(['Data-1','Data-3','Data-5','Data-10','Data-20'],loc="upper left")
            filename1 = dirname+'/'+'Accmulative_IC_Alpha' + str(i) + '.jpg'
            fig1 = plt.gcf()
            fig1.savefig(filename1, dpi = 300)
            plt.show()

            IC_list1 = self.getIC(self.date_list, self.Sure_data1, i)
            IC_list3 = self.getIC(self.date_list, self.Sure_data3, i)
            IC_list5 = self.getIC(self.date_list, self.Sure_data5, i)
            IC_list10 = self.getIC(self.date_list, self.Sure_data10, i)
            IC_list20 = self.getIC(self.date_list, self.Sure_data20, i)
            IC_mean = pd.DataFrame({"n = 1":IC_list1.groupby('year')["IC"].apply(lambda x: x.mean()),
                                    "n = 3":IC_list3.groupby('year')["IC"].apply(lambda x: x.mean()),
                                    "n = 5":IC_list5.groupby('year')["IC"].apply(lambda x: x.mean()),
                                    "n = 10":IC_list10.groupby('year')["IC"].apply(lambda x: x.mean()),
                                    "n = 20":IC_list20.groupby('year')["IC"].apply(lambda x: x.mean())})
            print(IC_mean)
            IC_mean.plot()
            plt.title('IC from 2010 to 2018 (Yearly)', color = '#6D6D6D', fontsize = 18)
            plt.legend(['Data-1','Data-3','Data-5','Data-10','Data-20'],loc="upper left")
            filename2 = dirname+'/'+'Yearly_IC_Alpha' + str(i) + '.jpg'
            fig2 = plt.gcf()
            fig2.savefig(filename2, dpi = 300)
            plt.show()

            IR = pd.DataFrame({"n = 1":IC_list1.groupby('year')["IC"].apply(lambda x: x.mean()) / IC_list1.groupby('year')["IC"].apply(lambda x: x.std()),
                                "n = 3":IC_list3.groupby('year')["IC"].apply(lambda x: x.mean()) / IC_list3.groupby('year')["IC"].apply(lambda x: x.std()),
                                "n = 5":IC_list5.groupby('year')["IC"].apply(lambda x: x.mean()) / IC_list5.groupby('year')["IC"].apply(lambda x: x.std()),
                                "n = 10":IC_list10.groupby('year')["IC"].apply(lambda x: x.mean()) / IC_list10.groupby('year')["IC"].apply(lambda x: x.std()),
                                "n = 20":IC_list20.groupby('year')["IC"].apply(lambda x: x.mean()) / IC_list20.groupby('year')["IC"].apply(lambda x: x.std())})
            print(IR)
            IR.plot()
            plt.title('IR from 2010 to 2018 (Yearly)', color = '#6D6D6D', fontsize = 18)
            plt.legend(['Data-1','Data-3','Data-5','Data-10','Data-20'],loc="upper left")
            filename3 = dirname+'/'+'Yearly_IR_Alpha' + str(i) + '.jpg'
            fig3 = plt.gcf()
            fig3.savefig(filename3, dpi = 300)
            plt.show()
            
            
    def get_new_yearly_corr(self):
        self.Sure_data1['year']=pd.to_datetime(self.Sure_data1['trade_date'],format='%Y-%m-%d').dt.strftime('%Y')
        corr1 = self.Sure_data1.groupby('year')[self.Sure_data1.columns[4:-2]].apply(lambda x: x.corr())
        self.pure_alpha_data = self.Sure_data1[self.Sure_data1.columns[4:-2]]
        self.corr2 = self.pure_alpha_data.corr()
        # self.corr2 = pd.DataFrame(self.corr2)
        # print(type(self.corr2))
        return corr1, self.corr2


    def hot_plot(self):
        cmap = cm.Blues
        # cmap = cm.hot_r
        fig=plt.figure()
        ax=fig.add_subplot(111)
        map = ax.imshow(self.corr2, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.title('correlation coefficient--headmap')
        ax.set_yticks(range(len(self.corr2.columns)))
        ax.set_yticklabels(self.corr2.columns)
        ax.set_xticks(range(len(self.corr2)))
        ax.set_xticklabels(self.corr2.columns)
        plt.colorbar(map)
        plt.show()

    def kmo(self):
        # KMO测度
        corr_inv = np.linalg.inv(self.corr2)
        nrow_inv_corr, ncol_inv_corr = self.corr2.shape
        A = np.ones((nrow_inv_corr, ncol_inv_corr))
        for i in range(0, nrow_inv_corr, 1):
            for j in range(i, ncol_inv_corr, 1):
                A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                A[j, i] = A[i, j]
        self.corr2 = np.asarray(self.corr2)
        kmo_num = np.sum(np.square(self.corr2)) - np.sum(np.square(np.diagonal(A)))
        kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
        self.kmo_value = kmo_num / kmo_denom
        print('KMO value is: ', self.kmo_value)
        
    def bart(self):
        corr = self.pure_alpha_data.corr().values
        # 计算结果有问题
        bart = st.bartlett(*corr)
        detCorr = np.linalg.det(corr)
        n = len(self.pure_alpha_data)
        p = len(self.pure_alpha_data.columns)
        statistic = -math.log(detCorr) * (n - 1 - (2 * p + 5) / 6)
        print("The statistic is: ", statistic)
        df = p * (p - 1) / 2
        # 双侧概率
        pval = (1.0 - st.chi2.cdf(statistic, df)) * 2
        print("The p-value is: ", pval)
        return bart

    def get_eig_data(self):
        # 求特征值和特征向量
        eig_value, eigvector = nlg.eig(self.corr2)  # 求矩阵R的全部特征值，构成向量
        self.eig_val = pd.DataFrame()
        self.eig_val['names'] = pd.DataFrame(self.corr2).columns
        self.eig_val['eig_value'] = eig_value
        self.eig_val.sort_values('eig_value', ascending=False, inplace=True)
        print("特征值：",self.eig_val)
        self.eig_vectors=pd.DataFrame(eigvector)
        self.eig_vectors.columns = pd.DataFrame(self.corr2).columns
        self.eig_vectors.index = pd.DataFrame(self.corr2).columns
        print("特征向量",self.eig_vectors)

    def t_test(self):
        # create the dummy matrix
        # self.merge_data()
        # self.factors = pd.read_csv('/home/candidate/cand3/LightGBM_trial/data/Result.csv', index_col=[0])  # 用已经读好的数据
        df = pd.read_csv('/home/candidate/cand3/factor_evaluation/data/industry_id.csv', index_col=[0])
        df['one']=1
        df_wide = df.pivot(index='stock_id',columns='industry_id',values='one')
        df_wide = df_wide.fillna(0)
        df_wide.reset_index(inplace=True)
        self.industry_dummy = df_wide
        # get the regression matrix
        df_new = self.factors.merge(self.industry_dummy, on='stock_id', how='left')
        df_new['pct_chg']=df_new.groupby('stock_id')['CLOSE'].apply(lambda x: x/x.shift(1)-1)
        self.factors=df_new
        self.factors.to_csv('/home/candidate/cand2/output/191_factors.csv')
        df_regress = df_new.drop(['VWAP','CLOSE'],axis=1)
        df_regress[[i for i in range(1,self.num_alpha+1)]]=df_regress.groupby('stock_id')[[i for i in range(1,self.num_alpha+1)]].apply(lambda x: x.shift(1))
        self.reg_data = df_regress
        self.reg_data.reset_index(drop=True,inplace=True)
        # regression
        self.total_t_test_table = pd.DataFrame(index = ['abs_mean', 'pct_>2', 't-values_statistic', 't-values_pvalues', 'abs_mean / std'])
        for i in range(1, self.num_alpha + 1):
            df_reg = copy.deepcopy(self.reg_data)
            reg_data_i = df_reg[df_reg.columns[-31:-1]]
            reg_data_i['alpha'] = df_reg[i]  # change to i
            reg_data_i['y'] = df_reg['pct_chg']
            reg_data_i['trade_date'] = df_reg['trade_date']
            reg_data_i.fillna(0,inplace = True)
            reg_data_i.replace([np.inf,-np.inf],[10000,-10000])
            # self.reg = reg_data_i
            result_t = reg_data_i.groupby('trade_date')[reg_data_i.columns[:-1]].apply(lambda x: sm.OLS(x['y'], x[x.columns[:-1]]).fit().tvalues)
            result_para = reg_data_i.groupby('trade_date')[reg_data_i.columns[:-1]].apply(lambda x: sm.OLS(x['y'], x[x.columns[:-1]]).fit().params)
            df = abs(result_t['alpha']).dropna()
            df_raw = result_t['alpha'].dropna()
            self.total_t_test_table[i] = [df.mean(), (df > 2).sum(axis = 0) / len(df), st.ttest_1samp(result_para['alpha'],0)[0], st.ttest_1samp(result_para['alpha'],0)[1], df.mean() / df_raw.std()]
        self.total_t_test_table.to_csv('/home/candidate/cand2/output/t-test.csv')


# YYJ test two
class factor_portfolio_sort(testFactors):
    def __init__(self,filePath, startdate, enddate, num_alpha, num_stock_file, wgt, portfolio_number=10, fee_rate=0.03):
        super(factor_portfolio_sort,self).__init__(filePath, startdate, enddate, num_alpha, num_stock_file) # inherit from Father Class
        """
        wgt: a dataframe with cols: stock_id, trade_date, value which daily weight is caculated based on. 
        Notice that factor and wgt at time t should contain information at time t, and we will deal with time together later.
        portfolio_number: the number of portfolios, by default=10
        """
        self.fee_rate=fee_rate
        self.portfolio_number=portfolio_number
        wgt['trade_date']=pd.to_datetime(wgt['trade_date']).dt.strftime('%Y-%m-%d')
        self.wgt=wgt
        self.wgt_name=wgt.columns[2]
        # sort values
    def data_processing(self):
        stock_ret_all=self.factors
        stock_ret_all=stock_ret_all[(stock_ret_all['trade_date']>=self.startdate)&(stock_ret_all['trade_date']<=self.enddate)]
        stock_ret_all=stock_ret_all.sort_values(by=['stock_id','trade_date'])
        stock_ret_all.reset_index(inplace=True,drop=True)
        stock_ret_all=stock_ret_all.merge(self.wgt,on=['stock_id','trade_date'],how='left')
        self.stock_ret_all=stock_ret_all
    def run_class(self,i):
        fee_rate=self.fee_rate
        factor_name='alpha'+str(i)
        data=self.stock_ret_all
        data=data[['stock_id','trade_date','pct_chg',self.wgt_name,i]]
        data.columns=['stock_id','trade_date','pct_chg',self.wgt_name,factor_name]
        data.reset_index(drop=True,inplace=True)
        # let return_t correponds to wgt_t-1 and alpha_t-1
        data[self.wgt_name]=data.groupby('stock_id')[self.wgt_name].apply(lambda x: x.shift(1))
        data['alpha'+str(i)]=data.groupby('stock_id')['alpha'+str(i)].apply(lambda x: x.shift(1))
        data.dropna(inplace=True)
        data.reset_index(inplace=True,drop=True)
        data=self.portfolio_data(data,factor_name)
        portfolios_cumret=self.portfolios_cumret_turnover(fee_rate=0,data=data)[0]
        portfolios_cumret_fee,portfolios_turnover_fee=self.portfolios_cumret_turnover(fee_rate=fee_rate,data=data)
        print('cumret without and with fee done')
        long_short_portfolio=pd.DataFrame()
        long_short_portfolio['1_no_fee']=portfolios_cumret[1]
        long_short_portfolio[str(self.portfolio_number)+'_no_fee']=portfolios_cumret[self.portfolio_number]
        long_short_portfolio['1_with_fee']=portfolios_cumret_fee[1]
        long_short_portfolio[str(self.portfolio_number)+'_with_fee']=portfolios_cumret_fee[self.portfolio_number]
        long_short_portfolio['cumret_no_fee']=long_short_portfolio[str(self.portfolio_number)+'_no_fee']-long_short_portfolio['1_no_fee']
        long_short_portfolio['cumret_with_fee']=long_short_portfolio['cumret_no_fee']\
            -(long_short_portfolio['1_no_fee']-long_short_portfolio['1_with_fee'])\
                -(long_short_portfolio[str(self.portfolio_number)+'_no_fee']-long_short_portfolio[str(self.portfolio_number)+'_with_fee'])
        long_short_portfolio.index=portfolios_cumret.index
        portfolios_cumret.to_csv('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+'cumret of 10 porfolios without transaction fee.csv')
        portfolios_cumret_fee.to_csv('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+'cumret of 10 porfolios with transaction fee.csv')
        portfolios_turnover_fee.to_csv('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+'turnover rate of 10 porfolios with transaction fee.csv')
        long_short_portfolio.to_csv('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+'cumret of long-short strategy without and with transaction fee.csv')
        print('csv for portfolio sort saved')
        plt.figure()
        portfolios_cumret.plot(legend=True,title='cumret of '+str(self.portfolio_number)+' porfolios without transaction fee')
        plt.legend(loc=2)
        myfig = plt.gcf() 
        myfig.savefig('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+'cumret of 10 porfolios without transaction fee.jpg',dpi=300)
        plt.figure()
        long_short_portfolio['cumret_with_fee'].plot(legend=True,title='cumret for long short strategy with transaction fee')
        myfig = plt.gcf()
        myfig.savefig('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+':cumret for long short strategy with transaction fee.jpg',dpi=300)
        plt.figure()
        long_short_portfolio['cumret_no_fee'].plot()
        (long_short_portfolio['1_no_fee']-long_short_portfolio['1_with_fee']).plot(legend='fee for portfolio 1')
        (long_short_portfolio[str(self.portfolio_number)+'_no_fee']-long_short_portfolio[str(self.portfolio_number)+'_with_fee']).plot()
        plt.legend(['cumret_no_fee','fee for portfolio 1','fee for portfolio 10'])
        plt.title('cumret for long short strategy without transaction fee')
        myfig = plt.gcf() 
        myfig.savefig('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+':cumret for long short strategy without transaction fee.jpg',dpi=300)
        plt.figure()
        portfolios_turnover_fee[self.portfolio_number].plot(legend=True,title='turnover rate for portfolio with highest factor value')
        myfig = plt.gcf()
        myfig.savefig('/home/candidate/cand2/output/'+'Alpha'+str(i)+'/'+':turnover rate for portfolio with highest factor value.jpg',dpi=300)
        return 
    def qcut(self,series, n):
        """
        assign values to differnt bins according to "values", which means that series with same number will be assigned to the same 
            middle portfolio.
        series: series to be allocated
        n: number of portfolios
        return: portfoilo number
        """
        edges = pd.Series([float(i) / n for i in range(n + 1)])
        f = lambda x: (edges >= x).argmax()
        port_number=series.rank(pct=1).apply(f)
        return port_number
    def portfolio_data(self,data,factor_name):
        """
        generate 10 portfolios and calculate wight in each portfolio
        return: nothing
        """
        # Categories (10, int64): [1 < 2 < 3 < 4 ... 7 < 8 < 9 < 10]
        data['port_category']=data.groupby('trade_date')[factor_name].apply(lambda x: self.qcut(x,self.portfolio_number))
        data['wgt']=data.groupby(['trade_date','port_category'])[self.wgt_name].apply(lambda x: x/sum(x))
        return data
    def cumret_turnover(self,df,fee_rate):
        """
        df: data for specific portfolio, including trade_date, stock_id, wgt, pct_chg
        fee_rate: fee rate per transaction
        short: if short=True, the the initial capital is set to be -1, and cumret will also be negative
        return: 2 dataframes with cols=['trade_date','cumret'] and cols=['trade_date','turnover_rate']
        """
        wgt_pivot=df.pivot(index='trade_date',columns='stock_id',values='wgt').fillna(0)
        ret_pivot=df.pivot(index='trade_date',columns='stock_id',values='pct_chg').fillna(0)
        cumret_pivot=wgt_pivot.copy(deep=True) # assument the initial capital for every portfolio is 1
        turnover_rate=pd.DataFrame(columns=['trade_date','turnover_rate'])
        turnover_rate['trade_date']=cumret_pivot.index
        turnover_rate['turnover_rate']=1 # initialize the turnover rate
        for i in range(1,cumret_pivot.shape[0]):
            if i!=cumret_pivot.shape[0]-1:
                close_capital_before_position_change=(ret_pivot.iloc[i]+1)*cumret_pivot.iloc[i-1]
                new_position=close_capital_before_position_change.sum()*wgt_pivot.iloc[i]
                position_change=np.abs(new_position-close_capital_before_position_change)
                turnover_rate.iloc[i,1]=position_change.sum()/close_capital_before_position_change.sum()
                cumret_pivot.iloc[i]=new_position-position_change*fee_rate # minus transaction fee on new position
            else:
                cumret_pivot.iloc[i]=(ret_pivot.iloc[i]+1)*cumret_pivot.iloc[i-1]-np.abs(cumret_pivot.iloc[i-1])*fee_rate # close the position for last date
        cumret=pd.DataFrame(cumret_pivot.sum(axis=1),columns=['cumret']).reset_index()
        return cumret,turnover_rate
    def portfolios_cumret_turnover(self,fee_rate,data):
        """
        calculate cumret and turnover rate of 10 porfolios with specified transaction fee rate
        return: cumulative return and turnover rate for 10 portfolios
        """
        port_cumret,port_turnover=self.cumret_turnover(data[data['port_category']==1],fee_rate)
        for i in range(2,self.portfolio_number+1):
            df1,df2=self.cumret_turnover(data[data['port_category']==i],fee_rate)
            port_cumret=port_cumret.merge(df1,on='trade_date',how='outer')
            port_turnover=port_turnover.merge(df2,on='trade_date',how='outer')
            # since the merge mode is outer, trade_date can be in wrong order when the port_categories != 10 for every day
        port_cumret=port_cumret.sort_values(by=['trade_date']).reset_index(drop=True)
        port_turnover=port_turnover.sort_values(by=['trade_date']).reset_index(drop=True)
        port_cumret.columns=['trade_date']+list(range(1,self.portfolio_number+1))
        port_turnover.columns=['trade_date']+list(range(1,self.portfolio_number+1))
        port_cumret=port_cumret.set_index('trade_date')# row: trade_date, col: 10 portfolios' cumulative return
        port_turnover=port_turnover.set_index('trade_date')# row: trade_date, col: 10 portfolios' turnover rate
        self.port_cumret=port_cumret
        self.port_turnover=port_turnover
        return self.port_cumret,self.port_turnover
#%%
# A = testFactors()
# A.merge_data()
# A.t_test()
# A.hot_plot()
# A.kmo()
# A.bart()
# A.get_eig_data()
#%%
from time import *
begin_time = time()
wgt=pd.read_csv('/home/candidate/cand3/factor_evaluation/data/stock_value2.csv',index_col=[0])
Factor_portfolio_sort=factor_portfolio_sort(filePath = '/home/candidate/cand3/factor_evaluation/data/stock_hfq/', 
                                            startdate = '2010-01-01',
                                            enddate = '2018-12-31',
                                            num_alpha = 10, 
                                            num_stock_file = 20,
                                            wgt = wgt,
                                            portfolio_number = 10,
                                            fee_rate = 0.003)
Factor_portfolio_sort.merge_data()
Factor_portfolio_sort.get_factors()
Factor_portfolio_sort.get_IC_Accmulative_IR_data()
corr_2010_2018, corr_total_times = Factor_portfolio_sort.get_new_yearly_corr()
corr_2010_2018.to_csv('/home/candidate/cand2/output/corr_2010_2018.csv')
corr_total_times.to_csv('/home/candidate/cand2/output/corr_total_times.csv')
Factor_portfolio_sort.t_test()
Factor_portfolio_sort.data_processing()
num_alpha=Factor_portfolio_sort.num_alpha
# for i in range(1,num_alpha+1):
#     Factor_portfolio_sort.run_class(i) # can be improved later
rounds=range(1,num_alpha+1)
executor = ProcessPoolExecutor(40)
futures = executor.map(Factor_portfolio_sort.run_class,rounds)
for i in futures:
    print(i)
print('done')
end_time = time()
run_time = end_time-begin_time
print ('该循环程序运行时间：',run_time)
#%%
# df['one']=1
# df_wide=df.pivot(index='stock_id',columns='industry_id',values='one')
# df_wide=df_wide.fillna(0)
# df_wide.reset_index(inplace=True)
# # ...
# df_new=df.merge(df_wide,on='stock_id',how='left')
# df_new['pct_chg']=df_new.groupby('stock_id')['CLOSE'].apply(lambda x: x/x.shift(1)-1)
# df_regress=df_new.drop(['VWAP','CLOSE'],axis=1)
# df_regress[[i for i in range(1,A.num_alpha+1)]]=df_regress.groupby('stock_id')[[i for i in range(1,A.num_alpha+1)]].apply(lambda x: x.shift(1))