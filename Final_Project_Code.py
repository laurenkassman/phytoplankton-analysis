#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load Python modules
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display, Image
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import binomtest, probplot


# In[2]:


# Load dataset and manipulate data frames for analysis
df_PHYTOBYGROUP = pd.read_excel(
    io='phytoplankton_data.xlsx',
    sheet_name='PHYTOBYGROUP')
df_stations = pd.read_excel(
    io='phytoplankton_data.xlsx',
    sheet_name='stations')
df_stations = df_stations.rename(
    columns={'Station': 'STATION'})
df_phytoplankton = df_PHYTOBYGROUP.merge(
    right=df_stations,
    how='left',
    on='STATION')
def map_great_lakes(row):
    if row['STATION'][0:2] == 'ER':
        row['great_lake'] = 'Erie'
    elif row['STATION'][0:2] == 'HU':
        row['great_lake'] = 'Huron'
    elif row['STATION'][0:2] == 'MI':
        row['great_lake'] = 'Michigan'
    elif row['STATION'][0:2] == 'ON':
        row['great_lake'] = 'Ontario'
    elif row['STATION'][0:2] == 'SU':
        row['great_lake'] = 'Superior'
    else:
        row['great_lake'] = 'Huron'
    return row
df_phytoplankton = df_phytoplankton.apply(
    func=map_great_lakes,
    axis=1)
df_phytoplankton['log_BAC'] = np.log(df_phytoplankton['BAC']+1)
df_phytoplankton['log_BAP'] = np.log(df_phytoplankton['BAP']+1)
df_phytoplankton['log_CHL'] = np.log(df_phytoplankton['CHL']+1)
df_phytoplankton['log_CHR'] = np.log(df_phytoplankton['CHR']+1)
df_phytoplankton['log_CRY'] = np.log(df_phytoplankton['CRY']+1)
df_phytoplankton['log_CYA'] = np.log(df_phytoplankton['CYA']+1)
df_phytoplankton['log_EUG'] = np.log(df_phytoplankton['EUG']+1)
df_phytoplankton['log_PYR'] = np.log(df_phytoplankton['PYR']+1)
df_phytoplankton['log_UNI'] = np.log(df_phytoplankton['UNI']+1)
df_phytoplankton['log_grand_total'] = np.log(df_phytoplankton['Grand Total']+1)


# In[3]:


# Print the first 3 rows of the final data frame
print(df_phytoplankton.head(3))


# In[4]:


# Print the counts of rows and columns for each data frame
rows_PHYTOBYGROUP, cols_PHYTOBYGROUP = df_PHYTOBYGROUP.shape
rows_stations, cols_stations = df_stations.shape
rows_phytoplankton, cols_phytoplankton = df_phytoplankton.shape
print(f'''
df_PHYTOBYGROUP has {rows_PHYTOBYGROUP} rows 
and {cols_PHYTOBYGROUP} columns.
''')
print(f'''
df_stations has {rows_stations} rows 
and {cols_stations} columns.
''')
print(f'''
df_phytoplankton has {rows_phytoplankton} rows 
and {cols_phytoplankton} columns.
''')


# In[5]:


# RQ1 OLS Regression Test
X_1 = df_phytoplankton['YEAR']
X_1 = sm.add_constant(X_1)
y_1 = df_phytoplankton['log_grand_total']
linear_model = sm.OLS(y_1, X_1)
linear_results = linear_model.fit()
print(linear_results.summary())


# In[6]:


# RQ1 OLS Regression Results
r_sq_linear = linear_results.rsquared
F_linear = linear_results.fvalue
p_value_F_linear = linear_results.f_pvalue
y_intercept_linear, slope_linear = linear_results.params
t_slope_linear = linear_results.tvalues.loc['YEAR']
p_slope_linear = linear_results.pvalues.loc['YEAR']
print(f'''
The coefficient of determination R^2 = {r_sq_linear:.3f}.
The F-test statistic for R^2 = {F_linear:.3f}.
The p-value for R^2 = {p_value_F_linear:.3f}.
The Simple Linear Regression slope parameter beta1 = {slope_linear:.3f}.
The t-test statistic for beta1 = {t_slope_linear:.3f}.
The p-value for beta1 = {p_slope_linear:.3f}.
''')


# In[7]:


# RQ1 Histogram of log grand total
fig_1_hist = px.histogram(
    data_frame=df_phytoplankton,
    x='log_grand_total',
    labels={'log_grand_total': 'log grand total'},
    title='Histogram of log total phytoplankton biovolume')
fig_1_hist.write_image('plot_1_hist.png')
display(Image(filename='plot_1_hist.png'))


# In[8]:


# RQ1 Q-Q plot of residuals
yhat_linear = y_intercept_linear+slope_linear*df_phytoplankton['YEAR']
resid_linear = df_phytoplankton['log_grand_total']-yhat_linear
resid_linear_sort = sorted(resid_linear)
qq_resid_linear = probplot(resid_linear_sort, dist='norm')
theo_quants_linear = qq_resid_linear[0][0]
sample_quants_linear = qq_resid_linear[0][1]
theo_slope_linear = qq_resid_linear[1][0]
theo_intercept_linear = qq_resid_linear[1][1]
fig_1_s_qq = px.scatter(
    x=theo_quants_linear,
    y=sample_quants_linear,
    labels={'x': 'Theoretical quantiles',                                           
            'y': 'Sample quantiles'},
    title='Q-Q plot of log total biovolume residuals by year')
fig_1_l_qq = px.line(
    x=theo_quants_linear,
    y=theo_intercept_linear+theo_slope_linear*theo_quants_linear,
    color_discrete_sequence=['red'])
fig_1_s_qq.add_traces(fig_1_l_qq.data)
fig_1_s_qq.write_image('plot_1_qq.png')
display(Image(filename='plot_1_qq.png'))


# In[9]:


# RQ1 Plot of residuals vs. predicted values
fig_1_s_yhat = px.scatter(
    x=yhat_linear,
    y=resid_linear,
    labels={'x': 'Predicted values (yhat)',                                           
            'y': 'Residuals (y-yhat)'},
    title='Plot of residuals vs. predicted values' \
        ' for log total biovolume by year')
fig_1_s_yhat.add_hline(
    y=0,
    line_color='red')
fig_1_s_yhat.write_image('plot_1_yhat.png')
display(Image(filename='plot_1_yhat.png'))


# In[10]:


# RQ1 Durbin-Watson test for independence
dw_linear = durbin_watson(linear_results.resid)
print(f'The Durbin-Watson test statistic = {dw_linear:.3f}.')


# In[11]:


# RQ1 Scatter plot with estimated regression line
fig_1_s_final = px.scatter(
    data_frame=df_phytoplankton,
    x='YEAR',
    y='log_grand_total',
    labels={'YEAR': 'year',
            'log_grand_total': 'log grand total'},
    title='Plot of log total biovolume vs. year' \
        ' with estimated regression line')
fig_1_l_final = px.line(
    x=df_phytoplankton['YEAR'],
    y=y_intercept_linear+slope_linear*df_phytoplankton['YEAR'],
    color_discrete_sequence=['red'])
fig_1_s_final.add_traces(fig_1_l_final.data)
fig_1_s_final.update_layout(xaxis_type='category')
fig_1_s_final.write_image('plot_1_final.png')
display(Image(filename='plot_1_final.png'))


# In[12]:


# RQ2 ANOVA test
anova_model = ols(
    'log_grand_total ~ C(great_lake)', 
    data=df_phytoplankton).fit() 
anova_results = anova_lm(anova_model)
print(anova_results)


# In[13]:


# RQ2 ANOVA results
F_anova = anova_results.iloc[0]['F']
p_value_F_anova = anova_results.iloc[0]['PR(>F)']
print(f'''
The F-test statistic for ANOVA = {F_anova:.3f}.
The p-value for ANOVA = {p_value_F_anova:.3f}.
''')


# In[14]:


# RQ2 Tukey HSD test
tukey_results = pairwise_tukeyhsd(
    endog=df_phytoplankton['log_grand_total'],
    groups=df_phytoplankton['great_lake'],
    alpha=0.01
)
print(tukey_results)


# In[15]:


# RQ2 Counts by Great Lake
counts_lake = df_phytoplankton['great_lake'].value_counts()
total_lake = sum(counts_lake)
print(counts_lake)
print(f'''
The total observation count across all 5 lakes = {total_lake}.
The number of rows in df_phytoplankton = {rows_phytoplankton}.
''')


# In[16]:


# RQ2 Q-Q plot of residuals
mean_anova = df_phytoplankton.groupby(
    'great_lake')['log_grand_total'].transform('mean')
resid_anova = df_phytoplankton['log_grand_total']-mean_anova
resid_anova_sort = sorted(resid_anova)
qq_resid_anova = probplot(resid_anova_sort, dist='norm')
theo_quants_anova = qq_resid_anova[0][0]
sample_quants_anova = qq_resid_anova[0][1]
theo_slope_anova = qq_resid_anova[1][0]
theo_intercept_anova = qq_resid_anova[1][1]
fig_2_s_qq = px.scatter(
    x=theo_quants_anova,
    y=sample_quants_anova,
    labels={'x': 'Theoretical quantiles',                                           
            'y': 'Sample quantiles'},
    title='Q-Q plot of log total biovolume residuals' \
        ' by Great Lake')
fig_2_l_qq = px.line(
    x=theo_quants_anova,
    y=theo_intercept_anova+theo_slope_anova*theo_quants_anova,
    color_discrete_sequence=['red'])
fig_2_s_qq.add_traces(fig_2_l_qq.data)
fig_2_s_qq.write_image('plot_2_qq.png')
display(Image(filename='plot_2_qq.png'))


# In[17]:


# RQ2 Standard deviation calculation
std_anova = df_phytoplankton.groupby(
    'great_lake')['log_grand_total'].agg('std')
range_anova = std_anova.max()-std_anova.min()
print(std_anova)
print('')
print(f'The range = {range_anova:.3f}.')


# In[18]:


# RQ2 Box plot of log total phytoplankton sample biovolume by Great Lake
fig_2_final = px.box(
    data_frame=df_phytoplankton,
    x='great_lake',
    y='log_grand_total',
    color='great_lake',
    labels={'great_lake': 'Great Lake',
            'log_grand_total': 'log grand total'},
    title='Box plots of log total biovolume by Great Lake')
fig_2_final.write_image('plot_2_final.png')
display(Image(filename='plot_2_final.png'))


# In[19]:


# RQ3 Multiple Logistic Regression test
X_3 = df_phytoplankton.iloc[:, [22,23,25,28]]
y_3 = df_phytoplankton['CRUISE']
X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(
    X_3, y_3, 
    test_size=0.3, 
    random_state=67)
logistic_model = LogisticRegression().fit(
    X_3_train, y_3_train)
y_3_pred = logistic_model.predict(X_3_test)
logistic_model_accuracy = accuracy_score(
    y_3_test, y_3_pred)
num_correct = np.sum(y_3_pred == y_3_test)
num_trials = len(y_3_test)
classes, counts = np.unique(
    y_3_test, 
    return_counts=True)
majority_class_count = np.max(counts)
NIR = majority_class_count/num_trials
logistic_test_result = binomtest(
    k=num_correct, 
    n=num_trials, 
    p=NIR, 
    alternative='greater')
print(logistic_test_result)


# In[20]:


# RQ3 Multiple Logistic Regression results
stat_binom = logistic_test_result.statistic
p_value_binom = logistic_test_result.pvalue
print(f'''
The No-Information Rate = {NIR:.3f}.
The Binomial test statistic = {stat_binom:.3f}
(which is the test accuracy of the classification model).
The Binomial p-value = {p_value_binom:.3f}.
''')


# In[21]:


# RQ3 Number of cruise outcomes
cruise_unique = df_phytoplankton['CRUISE'].unique()
print(cruise_unique)


# In[22]:


# RQ3 Counts by cruise
counts_cruise = df_phytoplankton['CRUISE'].value_counts()
total_cruise = sum(counts_cruise)
print(counts_cruise)
print(f'''
The total observation count across both cruises = {total_cruise}.
The number of rows in df_phytoplankton = {rows_phytoplankton}.''')


# In[23]:


# RQ3 Sample size of biovolume variables
zeros_log_bio = (df_phytoplankton==0).sum()[21:30]
zeros_log_bio = zeros_log_bio.sort_values()
print(zeros_log_bio)


# In[24]:


# RQ3 Scatter plot matrix of biovolume variables
fig_3_scatter = px.scatter_matrix(
    data_frame=df_phytoplankton, 
    dimensions=['log_BAC', 'log_BAP', 'log_CHL', 'log_CHR', 
                'log_CRY', 'log_CYA', 'log_PYR'],
    title='Scatter plots of biovolume across 7 groups')
fig_3_scatter.update_traces(showupperhalf=False, diagonal_visible=False)
fig_3_scatter.write_image('plot_3_scatter.png', width=800, height=800)
display(Image(filename='plot_3_scatter.png'))


# In[25]:


# RQ3 VIF to test multicollinearity
X_3_VIF = df_phytoplankton.iloc[:, [21,22,23,24,25,26,28]]
X_3_VIF = sm.add_constant(X_3_VIF)
df_VIF = pd.DataFrame({
    'Predictor': X_3_VIF.columns,
    'VIF': [variance_inflation_factor(X_3_VIF.values, i)
            for i in range(len(X_3_VIF.columns))]
})
print(df_VIF[1:])


# In[26]:


# RQ3 Descriptive statistics tables by cruise
df_phyto_spring = df_phytoplankton[df_phytoplankton['CRUISE']=='Spring']
df_phyto_spring = df_phyto_spring.iloc[:, [21,22,23,24,25,26,28]]
print('Spring summary:')
print(df_phyto_spring.describe())
df_phyto_summer = df_phytoplankton[df_phytoplankton['CRUISE']=='Summer']
df_phyto_summer = df_phyto_summer.iloc[:, [21,22,23,24,25,26,28]]
print('\nSummer summary:')
print(df_phyto_summer.describe())


# In[27]:


# RQ3 Box-Tidwell test for linearity in the logit
X_3_logit = df_phytoplankton.iloc[:, [21,22,23,24,25,26,28]]
y_3_logit = df_phytoplankton['CRUISE']
y_3_logit = y_3_logit.map(
    {'Spring': 0,
     'Summer': 1})
for col in X_3_logit.columns:
    X_3_logit[f'{col}_ln'] = X_3_logit[col] * np.log(X_3_logit[col]+1)
X_3_logit = sm.add_constant(X_3_logit)
logit_model = sm.Logit(y_3_logit, X_3_logit).fit()
print(logit_model.pvalues[8:])


# In[28]:


# RQ3 Confusion matrix for cruise classification
y_3_cm = confusion_matrix(y_3_test, y_3_pred, labels=classes)
fig_3_final = px.imshow(
    img=y_3_cm, x=classes, y=classes, text_auto=True,
    labels={'x': 'Predicted cruise',
            'y': 'True cruise',
            'color':'Count'},
    title='Confusion matrix for cruise classification')
fig_3_final.write_image('plot_3_final.png')
display(Image(filename='plot_3_final.png'))

