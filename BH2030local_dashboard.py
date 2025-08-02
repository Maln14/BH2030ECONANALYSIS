import streamlit as st
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="BH Economic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BahrainTimeSeriesAnalysis:
    def __init__(self, data):
        self.data = data
        self.projection_years = np.arange(2024, 2031)

    def identify_trend(self, ts_data, indicator_name):
        """Identify and analyze trends in time series data"""
        if len(ts_data) < 3:
            return {"trend": "Insufficient data", "strength": 0, "direction": "Unknown"}

        # Calculate trend using linear regression
        X = ts_data['Year'].values.reshape(-1, 1)
        y = ts_data['Value'].values

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)

        # Determine trend direction and strength
        if abs(slope) < 0.01:
            direction = "Stable"
        elif slope > 0:
            direction = "Increasing"
        else:
            direction = "Decreasing"

        # Trend strength based on R-squared
        if r2 > 0.8:
            strength = "Strong"
        elif r2 > 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"

        # Calculate volatility
        volatility = np.std(y) / np.mean(y) * 100 if np.mean(y) != 0 else 0

        # Identify cyclical patterns using autocorrelation
        if len(y) > 4:
            # Simple cyclical check - look for peaks and valleys
            peaks = 0
            valleys = 0
            for i in range(1, len(y)-1):
                if y[i] > y[i-1] and y[i] > y[i+1]:
                    peaks += 1
                elif y[i] < y[i-1] and y[i] < y[i+1]:
                    valleys += 1

            cyclical = "Yes" if (peaks + valleys) > len(y) * 0.3 else "No"
        else:
            cyclical = "Unknown"

        return {
            "trend": direction,
            "strength": strength,
            "slope": slope,
            "r_squared": r2,
            "volatility": volatility,
            "cyclical": cyclical,
            "avg_annual_change": slope,
            "total_change_percent": ((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0
        }

    def prepare_time_series_data(self, indicator_name, dataset_key='economic'):
        """Prepare time series data for a specific indicator"""
        df = self.data[dataset_key]

        if dataset_key == 'economic':
            filtered_data = df[df['Indicators (Current Prices)'].str.contains(indicator_name, na=False, case=False)]
            if not filtered_data.empty:
                ts_data = filtered_data[['Year', 'Value \nBD Million']].copy()
                ts_data.columns = ['Year', 'Value']
                ts_data = ts_data.sort_values('Year').reset_index(drop=True)
                ts_data['Value'] = pd.to_numeric(ts_data['Value'], errors='coerce')
                return ts_data.dropna()
        elif dataset_key == 'trade':
            filtered_data = df[df['Annual foreign trade statistics'].str.contains(indicator_name, na=False, case=False)]
            if not filtered_data.empty:
                ts_data = filtered_data[['Year', 'Value (Million BD)']].copy()
                ts_data.columns = ['Year', 'Value']
                ts_data = ts_data.sort_values('Year').reset_index(drop=True)
                ts_data['Value'] = pd.to_numeric(ts_data['Value'], errors='coerce')
                return ts_data.dropna()

        return pd.DataFrame()

    def prepare_fdi_data(self, sector_name, fdi_type='inward'):
        """Prepare FDI time series data for specific sector"""
        dataset_key = 'inward_fdi' if fdi_type == 'inward' else 'outward_fdi'
        df = self.data[dataset_key]

        if fdi_type == 'inward':
            filtered_data = df[df['Industry (ISIC Economic Activity)'].str.contains(sector_name, na=False, case=False)]
            if not filtered_data.empty:
                ts_data = filtered_data[['Year', 'Value.']].copy()
                ts_data.columns = ['Year', 'Value']
                ts_data = ts_data.sort_values('Year').reset_index(drop=True)
                ts_data['Value'] = pd.to_numeric(ts_data['Value'], errors='coerce')
                return ts_data.dropna()
        else:
            filtered_data = df[df['Industry (ISIC Economic Activity)'].str.contains(sector_name, na=False, case=False)]
            if not filtered_data.empty:
                ts_data = filtered_data[['Year', 'Value']].copy()
                ts_data = ts_data.sort_values('Year').reset_index(drop=True)
                ts_data['Value'] = pd.to_numeric(ts_data['Value'], errors='coerce')
                return ts_data.dropna()

        return pd.DataFrame()

    def linear_projection(self, ts_data):
        """Create linear regression projection"""
        if len(ts_data) < 3:
            return None, None, None

        X = ts_data['Year'].values.reshape(-1, 1)
        y = ts_data['Value'].values

        model = LinearRegression()
        model.fit(X, y)

        # Historical predictions
        historical_pred = model.predict(X)

        # Future predictions
        future_X = self.projection_years.reshape(-1, 1)
        future_pred = model.predict(future_X)

        # Calculate metrics
        r2 = r2_score(y, historical_pred)
        mae = mean_absolute_error(y, historical_pred)

        return future_pred, r2, mae

    def polynomial_projection(self, ts_data, degree=2):
        """Create polynomial regression projection"""
        if len(ts_data) < degree + 2:
            return None, None, None

        X = ts_data['Year'].values.reshape(-1, 1)
        y = ts_data['Value'].values

        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        poly_model.fit(X, y)

        # Historical predictions
        historical_pred = poly_model.predict(X)

        # Future predictions
        future_X = self.projection_years.reshape(-1, 1)
        future_pred = poly_model.predict(future_X)

        # Calculate metrics
        r2 = r2_score(y, historical_pred)
        mae = mean_absolute_error(y, historical_pred)

        return future_pred, r2, mae

    def exponential_smoothing_projection(self, ts_data):
        """Create exponential smoothing projection"""
        if len(ts_data) < 4:
            return None, None, None

        try:
            # Create time series with proper frequency
            ts_indexed = ts_data.set_index('Year')['Value']

            # Try different exponential smoothing models
            models = []

            # Simple exponential smoothing
            try:
                model_simple = ExponentialSmoothing(ts_indexed, trend=None, seasonal=None)
                fitted_simple = model_simple.fit()
                forecast_simple = fitted_simple.forecast(len(self.projection_years))
                models.append(('Simple', fitted_simple, forecast_simple))
            except:
                pass

            # Linear trend
            try:
                model_trend = ExponentialSmoothing(ts_indexed, trend='add', seasonal=None)
                fitted_trend = model_trend.fit()
                forecast_trend = fitted_trend.forecast(len(self.projection_years))
                models.append(('Trend', fitted_trend, forecast_trend))
            except:
                pass

            if not models:
                return None, None, None

            # Select best model based on AIC
            best_model = min(models, key=lambda x: x[1].aic)
            fitted_model = best_model[1]
            forecast = best_model[2]

            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            r2 = r2_score(ts_indexed.values, fitted_values.values)
            mae = mean_absolute_error(ts_indexed.values, fitted_values.values)

            return forecast.values, r2, mae

        except Exception as e:
            return None, None, None

    def create_projection_visualization(self, ts_data, indicator_name):
        """Create comprehensive projection visualization with trend analysis"""
        if ts_data.empty:
            st.warning(f"No data available for {indicator_name}")
            return

        # Perform trend analysis
        trend_analysis = self.identify_trend(ts_data, indicator_name)

        # Get projections from different models
        linear_proj, linear_r2, linear_mae = self.linear_projection(ts_data)
        poly_proj, poly_r2, poly_mae = self.polynomial_projection(ts_data)
        exp_proj, exp_r2, exp_mae = self.exponential_smoothing_projection(ts_data)

        # Create visualization
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=ts_data['Year'],
            y=ts_data['Value'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))

        # Add projections
        if linear_proj is not None:
            fig.add_trace(go.Scatter(
                x=self.projection_years,
                y=linear_proj,
                mode='lines+markers',
                name=f'Linear Projection (R²={linear_r2:.3f})',
                line=dict(color='red', dash='dash'),
                marker=dict(symbol='triangle-up')
            ))

        if poly_proj is not None:
            fig.add_trace(go.Scatter(
                x=self.projection_years,
                y=poly_proj,
                mode='lines+markers',
                name=f'Polynomial Projection (R²={poly_r2:.3f})',
                line=dict(color='green', dash='dot'),
                marker=dict(symbol='diamond')
            ))

        if exp_proj is not None:
            fig.add_trace(go.Scatter(
                x=self.projection_years,
                y=exp_proj,
                mode='lines+markers',
                name=f'Exponential Smoothing (R²={exp_r2:.3f})',
                line=dict(color='orange', dash='dashdot'),
                marker=dict(symbol='square')
            ))

        # Add vertical line at 2023 to separate historical and projected data
        fig.add_vline(x=2023.5, line_dash="solid", line_color="gray", 
                     annotation_text="Projection Start", annotation_position="top")

        fig.update_layout(
            title=f'{indicator_name} - Historical Data and 2030 Projections',
            xaxis_title='Year',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display trend analysis
        st.subheader(f"Trend Analysis for {indicator_name}")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Trend Direction", trend_analysis["trend"])
        with col2:
            st.metric("Trend Strength", trend_analysis["strength"])
        with col3:
            st.metric("Volatility", f"{trend_analysis['volatility']:.1f}%")
        with col4:
            st.metric("Cyclical Pattern", trend_analysis["cyclical"])

        # Additional trend insights
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Total Change", f"{trend_analysis['total_change_percent']:.1f}%")
        with col6:
            st.metric("Avg Annual Change", f"{trend_analysis['avg_annual_change']:.2f}")

        # Display projection summary table
        if any([linear_proj is not None, poly_proj is not None, exp_proj is not None]):
            st.subheader(f"2030 Projection Summary for {indicator_name}")

            summary_data = []
            if linear_proj is not None:
                summary_data.append({
                    'Model': 'Linear Regression',
                    '2030 Projection': f"{linear_proj[-1]:,.2f}",
                    'R-squared': f"{linear_r2:.3f}",
                    'Mean Absolute Error': f"{linear_mae:.2f}"
                })

            if poly_proj is not None:
                summary_data.append({
                    'Model': 'Polynomial Regression',
                    '2030 Projection': f"{poly_proj[-1]:,.2f}",
                    'R-squared': f"{poly_r2:.3f}",
                    'Mean Absolute Error': f"{poly_mae:.2f}"
                })

            if exp_proj is not None:
                summary_data.append({
                    'Model': 'Exponential Smoothing',
                    '2030 Projection': f"{exp_proj[-1]:,.2f}",
                    'R-squared': f"{exp_r2:.3f}",
                    'Mean Absolute Error': f"{exp_mae:.2f}"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

def create_vision_2030_dashboard(data):
    """Create comprehensive Vision 2030 analysis dashboard"""
    st.title("Bahrain Vision 2030 - Time Series Analysis & Projections")
    st.markdown("*Advanced forecasting models for key economic indicators towards Vision 2030 goals*")

    # Add official context from government and World Bank reports
    with st.expander("Official Vision 2030 Context & World Bank Outlook", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Government Vision 2030 Goals (Official):**
            - Reduce oil dependency to <20% of GDP - ACHIEVED (14% current)
            - Achieve sustainable 3-4% annual GDP growth - ON TRACK (3.5% in 2025)
            - Increase private sector employment to 70% - ACHIEVED (70% current rate)
            - Maintain positive trade balance - ACHIEVED (4.8% surplus)
            - Increase private investment to 25% of GDP - MONITORING
            """)

        with col2:
            st.markdown("""
            **World Bank Economic Analysis:**
            - GDP growth: 3.5% (2025), 3% medium-term target
            - Fiscal deficit narrowing: 8.4% (2024) → 7.7% (2025)
            - Economic structure: 86% non-oil, 14% oil sectors
            - Risks: Energy price volatility, deficit >8% by 2026-27
            - Recovery drivers: BAPCO refinery, Sitra expansion
            - New revenues: 10% VAT, 15% corporate tax (2025)
            """)

    st.markdown("---")

    # Initialize analysis class
    analyzer = BahrainTimeSeriesAnalysis(data)

    # Sidebar for indicator selection
    st.sidebar.header("Analysis Configuration")

    # Key economic indicators for Vision 2030
    economic_indicators = {
        'GDP Growth': 'GDP.*Growth',
        'Gross Domestic Product': 'Gross Domestic Product.*(?!Growth)',
        'Non-oil GDP': 'Non-oil sector GDP',
        'Oil GDP': 'Oil sector GDP',
        'GDP per Capita': 'GDP per capita',
        'Private Consumption': 'Private.*% of GDP',
        'Government Consumption': 'Government.*% of GDP',
        'Private Investment': 'Private.*% of GDP.*Formation'
    }

    trade_indicators = {
        'Total Exports': 'Total Exports',
        'Oil Exports': 'Oil Exports',
        'Non-oil Exports': 'Non.*oil Exports',
        'Total Imports': 'Total Imports',
        'Trade Balance': 'Trade Balance'
    }

    # Indicator category selection
    analysis_category = st.sidebar.selectbox(
        "Select Analysis Category",
        ["Economic Indicators", "Trade Statistics", "FDI Analysis", "Comparative Analysis"]
    )

    if analysis_category == "Economic Indicators":
        selected_indicator = st.sidebar.selectbox(
            "Select Economic Indicator",
            list(economic_indicators.keys())
        )

        # Perform analysis
        ts_data = analyzer.prepare_time_series_data(
            economic_indicators[selected_indicator], 
            'economic'
        )
        analyzer.create_projection_visualization(ts_data, selected_indicator)

        # Additional insights
        if not ts_data.empty and len(ts_data) > 1:
            st.subheader("Historical Trends Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                avg_growth = ((ts_data['Value'].iloc[-1] / ts_data['Value'].iloc[0]) ** (1/(len(ts_data)-1)) - 1) * 100
                st.metric("Average Annual Growth Rate", f"{avg_growth:.2f}%")

            with col2:
                latest_value = ts_data['Value'].iloc[-1]
                st.metric("Latest Value", f"{latest_value:,.2f}")

            with col3:
                volatility = ts_data['Value'].std() / ts_data['Value'].mean() * 100
                st.metric("Coefficient of Variation", f"{volatility:.2f}%")

    elif analysis_category == "Trade Statistics":
        selected_indicator = st.sidebar.selectbox(
            "Select Trade Indicator",
            list(trade_indicators.keys())
        )

        # Perform analysis
        ts_data = analyzer.prepare_time_series_data(
            trade_indicators[selected_indicator], 
            'trade'
        )
        analyzer.create_projection_visualization(ts_data, selected_indicator)

        # Trade-specific insights
        if not ts_data.empty and len(ts_data) > 1:
            st.subheader("Trade Performance Analysis")

            col1, col2 = st.columns(2)

            with col1:
                total_change = ((ts_data['Value'].iloc[-1] / ts_data['Value'].iloc[0]) - 1) * 100
                st.metric("Total Change Since Start", f"{total_change:.1f}%")

            with col2:
                recent_trend = ((ts_data['Value'].iloc[-1] / ts_data['Value'].iloc[-2]) - 1) * 100 if len(ts_data) > 1 else 0
                st.metric("Recent Year Change", f"{recent_trend:.1f}%")

    elif analysis_category == "FDI Analysis":
        st.subheader("Foreign Direct Investment Analysis")

        # FDI type selection
        fdi_type = st.sidebar.selectbox(
            "Select FDI Type",
            ["Inward FDI", "Outward FDI"]
        )

        # Get available sectors
        if fdi_type == "Inward FDI":
            sectors = data['inward_fdi']['Industry (ISIC Economic Activity)'].unique()
        else:
            sectors = data['outward_fdi']['Industry (ISIC Economic Activity)'].unique()

        selected_sector = st.sidebar.selectbox(
            "Select Sector",
            sorted(sectors)
        )

        # Perform FDI analysis
        fdi_data = analyzer.prepare_fdi_data(
            selected_sector, 
            'inward' if fdi_type == "Inward FDI" else 'outward'
        )

        if not fdi_data.empty:
            analyzer.create_projection_visualization(fdi_data, f"{fdi_type} - {selected_sector}")

            # FDI-specific insights
            st.subheader("FDI Performance Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                latest_value = fdi_data['Value'].iloc[-1]
                st.metric("Latest Value (Million BD)", f"{latest_value:.2f}")

            with col2:
                if len(fdi_data) > 1:
                    growth_rate = ((fdi_data['Value'].iloc[-1] / fdi_data['Value'].iloc[0]) ** (1/(len(fdi_data)-1)) - 1) * 100
                    st.metric("CAGR", f"{growth_rate:.2f}%")

            with col3:
                volatility = fdi_data['Value'].std() / fdi_data['Value'].mean() * 100 if fdi_data['Value'].mean() != 0 else 0
                st.metric("Volatility", f"{volatility:.1f}%")
        else:
            st.warning(f"No data available for {selected_sector} in {fdi_type}")

# Cache data loading function
@st.cache_data
def load_data():
    """Load all datasets from CSV files"""
    try:
        import os
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Function to create full path
        def get_csv_path(filename):
            return os.path.join(script_dir, filename)

        # First, read all CSV files individually with full paths
        economic_df = pd.read_csv(get_csv_path('BHRGenEconIndic.csv'))
        employment_sector_df = pd.read_csv(get_csv_path('BHEmploybySec.csv'))
        employment_rates_df = pd.read_csv(get_csv_path('BHEmployRates.csv'))
        trade_df = pd.read_csv(get_csv_path('BHTradeStats.csv'))
        unemployment_age_df = pd.read_csv(get_csv_path('BHUnempAge&Gender.csv'))
        unemployment_rates_df = pd.read_csv(get_csv_path('BHUnemployRates.csv'))
        agricultural_df = pd.read_csv(get_csv_path('ECONIndicAgricultral.csv'))
        inward_fdi_df = pd.read_csv(get_csv_path('InwardFDI.csv'))
        outward_fdi_df = pd.read_csv(get_csv_path('OutwardFDI.csv'))

        # Then create the dictionary after successful loading
        loaded_data = {
            'economic': economic_df,
            'employment_sector': employment_sector_df,
            'employment_rates': employment_rates_df,
            'trade': trade_df,
            'unemployment_age': unemployment_age_df,
            'unemployment_rates': unemployment_rates_df,
            'agricultural': agricultural_df,
            'inward_fdi': inward_fdi_df,
            'outward_fdi': outward_fdi_df
        }

        return loaded_data

    except FileNotFoundError as e:
        st.error(f"CSV file not found: {e}")
        return None
    except Exception as error:
        st.error(f"Error loading data: {error}")
        return None

def create_gdp_analysis(economic_data):
    """Create GDP growth and composition visualizations"""
    st.subheader("GDP Growth and Composition Analysis")

    # Add GDP context from official reports
    st.info("""
    **World Bank GDP Outlook**: Growth projected to accelerate to 3.5% in 2025 driven by BAPCO refinery completion. 
    Medium-term GDP stabilizing around 3% with robust non-hydrocarbon sector expansion, particularly from Sitra oil refinery.
    Non-oil sector currently represents 86% of the economy.
    """)

    # Filter data to find GDP-related indicators
    gdp_data = economic_data[economic_data['Indicators (Current Prices)'].str.contains('GDP|Gross Domestic Product', na=False)]

    # Create two columns for side-by-side charts
    column1, column2 = st.columns(2)

    # First column: GDP Growth over time
    with column1:
        gdp_growth = gdp_data[gdp_data['Indicators (Current Prices)'].str.contains('GDP.*Growth', na=False)]

        if not gdp_growth.empty:
            # Create line chart for GDP growth
            growth_chart = px.line(
                gdp_growth, 
                x='Year', 
                y='Value \nBD Million', 
                color='Indicators (Current Prices)',
                title='GDP Growth Rate Over Time'
            )
            growth_chart.update_layout(
                xaxis_title="Year", 
                yaxis_title="Growth Rate (%)"
            )
            st.plotly_chart(growth_chart, use_container_width=True)

    # Second column: Oil vs Non-oil GDP composition
    with column2:
        oil_gdp = economic_data[economic_data['Indicators (Current Prices)'].str.contains('Oil sector GDP', na=False)]
        nonoil_gdp = economic_data[economic_data['Indicators (Current Prices)'].str.contains('Non-oil sector GDP', na=False)]

        if not oil_gdp.empty and not nonoil_gdp.empty:
            # Combine oil and non-oil data
            composition_data = pd.DataFrame({
                'Year': oil_gdp['Year'].values,
                'Oil Sector': oil_gdp['Value \nBD Million'].values,
                'Non-Oil Sector': nonoil_gdp['Value \nBD Million'].values
            })

            # Create area chart for composition
            composition_chart = px.area(
                composition_data, 
                x='Year', 
                y=['Oil Sector', 'Non-Oil Sector'],
                title='GDP Composition: Oil vs Non-Oil Sectors'
            )
            st.plotly_chart(composition_chart, use_container_width=True)

def create_employment_analysis(employment_data, unemployment_data):
    """Create employment and unemployment visualizations"""
    st.subheader("Employment and Labor Market Analysis")

    # Add employment context
    st.info("""
    **Labor Market Status**: Employment rates around 70% with overall unemployment at 1%. 
    However, female youth unemployment remains high at 12.4%, indicating need for targeted 
    policies to increase female labor force participation.
    """)

    # Create two columns for charts
    column1, column2 = st.columns(2)

    # First column: Employment by sector and gender
    with column1:
        if not employment_data.empty:
            # Group data by sector and gender, sum recruitment numbers
            sector_summary = employment_data.groupby(['Sector', 'Sex'])['Recruitment'].sum().reset_index()

            # Create grouped bar chart
            employment_chart = px.bar(
                sector_summary, 
                x='Sector', 
                y='Recruitment', 
                color='Sex',
                title='Employment by Sector and Gender',
                barmode='group'
            )
            employment_chart.update_xaxes(tickangle=45)
            st.plotly_chart(employment_chart, use_container_width=True)

    # Second column: Unemployment rates by gender over time
    with column2:
        if not unemployment_data.empty:
            # Filter for unemployment rate data
            unemployment_rates = unemployment_data[unemployment_data['Indicator'].str.contains('Rate', na=False)]

            # Create line chart for unemployment trends
            unemployment_chart = px.line(
                unemployment_rates, 
                x='Year', 
                y='Value', 
                color='Sex',
                title='Unemployment Rates by Gender Over Time'
            )
            unemployment_chart.update_layout(yaxis_title="Unemployment Rate (%)")
            st.plotly_chart(unemployment_chart, use_container_width=True)

def create_trade_analysis(trade_data):
    """Analyze trade patterns and export diversification"""
    st.subheader("Trade Statistics Analysis")

    # Add official trade context
    st.info("""
    **Official Trade Outlook**: Current account surplus of BHD 858.0 million (4.8% of GDP) in 2024, 
    with 9.6% increase in services exports. Trade balance maintains 7.6% of GDP surplus despite oil price volatility.
    """)

    # Create two columns for charts
    column1, column2 = st.columns(2)

    # First column: Oil vs Non-oil exports comparison
    with column1:
        # Filter export data
        exports_data = trade_data[trade_data['Annual foreign trade statistics'].str.contains('Exports', na=False)]
        oil_exports = exports_data[exports_data['Annual foreign trade statistics'].str.contains('Oil Exports', na=False)]
        nonoil_exports = exports_data[exports_data['Annual foreign trade statistics'].str.contains('Non_oil Exports|Non-oil Exports', na=False)]

        if not oil_exports.empty and not nonoil_exports.empty:
            # Create a clean stacked area chart
            export_chart = go.Figure()

            # Merge data on year for proper stacking
            merged_exports = pd.merge(
                oil_exports[['Year', 'Value (Million BD)']], 
                nonoil_exports[['Year', 'Value (Million BD)']], 
                on='Year', 
                suffixes=('_oil', '_nonoil'),
                how='outer'
            ).fillna(0).sort_values('Year')

            # Add stacked areas with smooth curves
            export_chart.add_trace(go.Scatter(
                x=merged_exports['Year'],
                y=merged_exports['Value (Million BD)_nonoil'],
                mode='none',
                name='Non-Oil Exports',
                fill='tozeroy',
                fillcolor='rgba(46, 125, 50, 0.7)',
                line=dict(width=0, shape='spline', smoothing=1.3),
                hovertemplate='<b>Non-Oil Exports</b><br>Year: %{x}<br>Value: %{y:,.0f} Million BD<extra></extra>'
            ))

            export_chart.add_trace(go.Scatter(
                x=merged_exports['Year'],
                y=merged_exports['Value (Million BD)_oil'] + merged_exports['Value (Million BD)_nonoil'],
                mode='none',
                name='Oil Exports',
                fill='tonexty',
                fillcolor='rgba(255, 152, 0, 0.7)',
                line=dict(width=0, shape='spline', smoothing=1.3),
                hovertemplate='<b>Oil Exports</b><br>Year: %{x}<br>Value: %{customdata:,.0f} Million BD<extra></extra>',
                customdata=merged_exports['Value (Million BD)_oil']
            ))

            # Update chart layout with modern styling
            export_chart.update_layout(
                title={
                    'text': 'Export Composition: Oil vs Non-Oil',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2C3E50'}
                },
                xaxis={
                    'title': 'Year',
                    'showgrid': True,
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'showline': True,
                    'linecolor': '#E0E0E0'
                },
                yaxis={
                    'title': 'Cumulative Export Value (Million BD)',
                    'showgrid': True,
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'showline': True,
                    'linecolor': '#E0E0E0'
                },
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400
            )

            st.plotly_chart(export_chart, use_container_width=True)

            # Add explanation text with better formatting
            st.markdown("""
            **What this chart shows:**```python
 This stacked area visualization displays Bahrain's export composition over time. 
            The green area represents non-oil exports, while the orange area shows oil exports stacked on top. 
            The total height at any point represents total export value, making it easy to see both 
            diversification progress and overall export performance.
            """)

            # Add key insights
            latest_year = merged_exports['Year'].max()
            latest_data = merged_exports[merged_exports['Year'] == latest_year].iloc[0]
            total_exports = latest_data['Value (Million BD)_oil'] + latest_data['Value (Million BD)_nonoil']
            nonoil_share = (latest_data['Value (Million BD)_nonoil'] / total_exports * 100) if total_exports > 0 else 0

            st.info(f"**Latest ({int(latest_year)}):** Non-oil exports represent {nonoil_share:.1f}% of total exports")

    # Second column: Trade balance analysis
    with column2:
        # Filter trade balance data
        trade_balance = trade_data[trade_data['Annual foreign trade statistics'].str.contains('Trade Balance', na=False)]

        if not trade_balance.empty:
            # Create color scheme: green for positive, red for negative
            bar_colors = []
            for value in trade_balance['Value (Million BD)']:
                if value >= 0:
                    bar_colors.append('green')
                else:
                    bar_colors.append('red')

            # Create bar chart for trade balance
            balance_chart = go.Figure(data=[
                go.Bar(
                    x=trade_balance['Year'],
                    y=trade_balance['Value (Million BD)'],
                    marker_color=bar_colors,
                    text=[f"{x:,.0f}" for x in trade_balance['Value (Million BD)']],
                    textposition='outside',
                    textfont=dict(size=10)
                )
            ])

            # Add zero reference line
            balance_chart.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

            # Update layout
            balance_chart.update_layout(
                title='Trade Balance Over Time',
                xaxis_title="Year",
                yaxis_title="Trade Balance (Million BD)",
                showlegend=False,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(balance_chart, use_container_width=True)

            # Add explanation text
            st.markdown("""
            **What this chart shows:** The trade balance represents the difference between exports and imports. 
            Green bars indicate a trade surplus (exports > imports), while red bars show a trade deficit. 
            A positive balance means Bahrain earned more from exports than it spent on imports.
            """)

def create_diversification_metrics(economic_data, trade_data):
    """Analyze economic diversification metrics"""
    st.subheader("Economic Diversification Metrics")

    # Add diversification context from reports
    st.info("""
    **Diversification Progress**: Non-oil GDP represents 86% of total economy (14% oil dependency - already achieving Vision 2030 target of <20%). 
    Key sectors: Financial Services (17.2% of GDP), Manufacturing (15.1%), Public Administration (8.5%). 
    Economic Recovery Plan (2022-2026) focuses on infrastructure and digital transformation. However, budget remains 64% reliant on hydrocarbon revenues.
    """)

    # Create two columns for metrics
    column1, column2 = st.columns(2)

    # First column: Non-oil sector contribution to GDP
    with column1:
        # Filter for non-oil GDP percentage data
        nonoil_contrib = economic_data[economic_data['Indicators (Current Prices)'].str.contains('Non-oil.*% of GDP', na=False)]

        if not nonoil_contrib.empty:
            # Create diversification progress chart
            diversification_chart = go.Figure()

            # Add main trend line with area fill
            diversification_chart.add_trace(go.Scatter(
                x=nonoil_contrib['Year'],
                y=nonoil_contrib['Value \nBD Million'],
                mode='lines+markers',
                name='Non-Oil GDP Share',
                fill='tozeroy',
                fillcolor='rgba(40, 167, 69, 0.2)',
                line=dict(color='green', width=4),
                marker=dict(size=8, color='green')
            ))

            # Add Vision 2030 target line (example: 80% target)
            diversification_chart.add_hline(
                y=80, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Vision 2030 Target (80%)",
                annotation_position="right"
            )

            # Update layout
            diversification_chart.update_layout(
                title='Non-Oil Sector Contribution to GDP',
                xaxis_title="Year",
                yaxis_title="Percentage of GDP (%)",
                yaxis=dict(range=[0, 100]),
                showlegend=False,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(diversification_chart, use_container_width=True)

            # Add explanation
            st.markdown("""
            **What this chart shows:** This measures how much of Bahrain's economy comes from non-oil sectors. 
            Higher percentages indicate greater economic diversification and reduced dependence on oil revenues. 
            The red dashed line shows Bahrain's Vision 2030 diversification target.
            """)

    # Second column: Export diversification progress
    with column2:
        # Get total exports and oil exports data
        total_exports = trade_data[trade_data['Annual foreign trade statistics'].str.contains('Total Exports', na=False)]
        oil_exports = trade_data[trade_data['Annual foreign trade statistics'].str.contains('Oil Exports', na=False)]

        if not total_exports.empty and not oil_exports.empty:
            # Combine data to calculate diversification
            merged_data = pd.merge(
                total_exports[['Year', 'Value (Million BD)']], 
                oil_exports[['Year', 'Value (Million BD)']], 
                on='Year', 
                suffixes=('_total', '_oil')
            )

            # Calculate non-oil share of exports
            merged_data['Non_oil_share'] = (
                (merged_data['Value (Million BD)_total'] - merged_data['Value (Million BD)_oil']) / 
                merged_data['Value (Million BD)_total'] * 100
            )

            # Create export diversification chart with clean styling
            export_div_chart = go.Figure()

            # Add smooth filled area without line artifacts
            export_div_chart.add_trace(go.Scatter(
                x=merged_data['Year'],
                y=merged_data['Non_oil_share'],
                mode='none',
                name='Non-Oil Export Share',
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.6)',
                line=dict(width=0, shape='spline', smoothing=1.3),
                hovertemplate='<b>Non-Oil Export Share</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>'
            ))

            # Add 50% benchmark line
            export_div_chart.add_hline(
                y=50, 
                line_dash="dot", 
                line_color="#F97316",
                line_width=2,
                annotation_text="Balanced Exports (50%)",
                annotation_position="right"
            )

            # Update layout with modern styling
            export_div_chart.update_layout(
                title={
                    'text': 'Export Diversification Progress',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2C3E50'}
                },
                xaxis={
                    'title': 'Year',
                    'showgrid': True,
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'showline': True,
                    'linecolor': '#E0E0E0'
                },
                yaxis={
                    'title': 'Non-Oil Share of Total Exports (%)',
                    'range': [0, 100],
                    'showgrid': True,
                    'gridcolor': 'rgba(0,0,0,0.1)',
                    'showline': True,
                    'linecolor': '#E0E0E0'
                },
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                showlegend=False,
                height=400
            )
            st.plotly_chart(export_div_chart, use_container_width=True)

            # Add explanation
            st.markdown("""
            **What this chart shows:** This tracks the percentage of Bahrain's exports that come from non-oil products. 
            A rising trend indicates successful export diversification, reducing reliance on oil exports. 
            The orange line at 50% represents a balanced export portfolio between oil and non-oil products.
            """)

def create_private_sector_analysis(economic_data):
    """Analyze private sector contribution to economy"""
    st.subheader("Private Sector Analysis")

    # Create two columns for analysis
    column1, column2 = st.columns(2)

    # First column: Private vs government consumption
    with column1:
        # Filter for consumption data
        private_consumption = economic_data[economic_data['Indicators (Current Prices)'].str.contains('Private.*% of GDP', na=False)]
        govt_consumption = economic_data[economic_data['Indicators (Current Prices)'].str.contains('Government.*% of GDP', na=False)]

        if not private_consumption.empty and not govt_consumption.empty:
            # Create comparison chart
            consumption_chart = go.Figure()

            # Add private consumption line
            consumption_chart.add_trace(go.Scatter(
                x=private_consumption['Year'], 
                y=private_consumption['Value \nBD Million'],
                mode='lines+markers', 
                name='Private Consumption'
            ))

            # Add government consumption line
            consumption_chart.add_trace(go.Scatter(
                x=govt_consumption['Year'], 
                y=govt_consumption['Value \nBD Million'],
                mode='lines+markers', 
                name='Government Consumption'
            ))

            consumption_chart.update_layout(
                title='Private vs Government Consumption (% of GDP)',
                yaxis_title="Percentage of GDP"
            )
            st.plotly_chart(consumption_chart, use_container_width=True)

    # Second column: Private investment trends
    with column2:
        # Filter for private investment data
        private_investment = economic_data[economic_data['Indicators (Current Prices)'].str.contains('Private.*% of GDP.*Formation', na=False)]

        if not private_investment.empty:
            # Create private investment chart
            investment_chart = px.line(
                private_investment, 
                x='Year', 
                y='Value \nBD Million',
                title='Private Investment (% of GDP)'
            )
            st.plotly_chart(investment_chart, use_container_width=True)

def create_fdi_analysis(inward_fdi, outward_fdi):
    """Create Foreign Direct Investment analysis visualizations"""
    st.subheader("Foreign Direct Investment Analysis")

    # Create two columns for FDI analysis
    column1, column2 = st.columns(2)

    # First column: Top inward FDI sectors
    with column1:
        if not inward_fdi.empty:
            # Find the most recent year in the data
            latest_year = inward_fdi['Year'].max()

            # Get top 5 sectors by value in latest year
            top_sectors = inward_fdi[inward_fdi['Year'] == latest_year].nlargest(5, 'Value.')

            # Create chart showing trends for top sectors
            sector_chart = go.Figure()

            for sector in top_sectors['Industry (ISIC Economic Activity)'].unique():
                sector_data = inward_fdi[inward_fdi['Industry (ISIC Economic Activity)'] == sector]

                # Shorten long sector names for readability
                display_name = sector[:30] + '...' if len(sector) > 30 else sector

                sector_chart.add_trace(go.Scatter(
                    x=sector_data['Year'], 
                    y=sector_data['Value.'],
                    mode='lines+markers',
                    name=display_name
                ))

            sector_chart.update_layout(
                title='Top 5 Inward FDI Sectors Over Time',
                xaxis_title='Year',
                yaxis_title='Value (Million BD)',
                showlegend=True
            )
            st.plotly_chart(sector_chart, use_container_width=True)

    # Second column: Outward FDI trends
    with column2:
        if not outward_fdi.empty:
            # Sum outward FDI by year
            outward_yearly = outward_fdi.groupby('Year')['Value'].sum().reset_index()

            # Create line chart for outward FDI
            outward_chart = px.line(
                outward_yearly, 
                x='Year', 
                y='Value',
                title='Total Outward FDI Over Time',
                markers=True
            )
            outward_chart.update_layout(yaxis_title='Value (Million BD)')
            st.plotly_chart(outward_chart, use_container_width=True)

    # FDI Balance Analysis section
    st.subheader("FDI Net Position Analysis")

    # Calculate net FDI position (inward minus outward)
    inward_yearly = inward_fdi.groupby('Year')['Value.'].sum().reset_index()
    outward_yearly = outward_fdi.groupby('Year')['Value'].sum().reset_index()

    # Rename columns for clarity
    inward_yearly.rename(columns={'Value.': 'Inward_FDI'}, inplace=True)
    outward_yearly.rename(columns={'Value': 'Outward_FDI'}, inplace=True)

    # Combine inward and outward data
    fdi_balance = pd.merge(inward_yearly, outward_yearly, on='Year', how='outer')
    fdi_balance = fdi_balance.fillna(0)  # Fill missing values with 0
    fdi_balance['Net_FDI'] = fdi_balance['Inward_FDI'] - fdi_balance['Outward_FDI']

    # Create comprehensive FDI balance chart
    fdi_balance_chart = go.Figure()

    # Add inward FDI bars (positive)
    fdi_balance_chart.add_trace(go.Bar(
        x=fdi_balance['Year'], 
        y=fdi_balance['Inward_FDI'], 
        name='Inward FDI', 
        marker_color='green'
    ))

    # Add outward FDI bars (negative for visual comparison)
    fdi_balance_chart.add_trace(go.Bar(
        x=fdi_balance['Year'], 
        y=-fdi_balance['Outward_FDI'], 
        name='Outward FDI', 
        marker_color='red'
    ))

    # Add net FDI position line
    fdi_balance_chart.add_trace(go.Scatter(
        x=fdi_balance['Year'], 
        y=fdi_balance['Net_FDI'], 
        mode='lines+markers', 
        name='Net FDI Position', 
        line=dict(color='blue', width=3)
    ))

    # Update chart layout
    fdi_balance_chart.update_layout(
        title='Bahrain FDI Position: Inward vs Outward Flows',
        xaxis_title='Year',
        yaxis_title='Value (Million BD)',
        barmode='relative'
    )
    st.plotly_chart(fdi_balance_chart, use_container_width=True)

def create_key_metrics_cards(data):
    """Create key performance indicator cards showing latest economic data"""
    st.subheader("Key Economic Indicators and Vision 2030 Progress")

    # Get the most recent year of data
    latest_year = data['economic']['Year'].max()
    latest_data = data['economic'][data['economic']['Year'] == latest_year]

    # Create four columns for key metrics
    column1, column2, column3, column4 = st.columns(4)

    # Column 1: GDP value
    with column1:
        # Find GDP data (excluding growth rates)
        gdp_data = latest_data[latest_data['Indicators (Current Prices)'].str.contains('Gross Domestic Product.*GDP.*(?!Growth)', na=False)]

        if not gdp_data.empty:
            gdp_value = gdp_data['Value \nBD Million'].iloc[0]
        else:
            gdp_value = 0

        st.metric("GDP (Million BD)", f"{gdp_value:,.0f}")

    # Column 2: Oil sector share
    with column2:
        # Find oil sector percentage of GDP
        oil_share_data = latest_data[latest_data['Indicators (Current Prices)'].str.contains('Oil sector.*% of GDP', na=False)]

        if not oil_share_data.empty:
            oil_share = oil_share_data['Value \nBD Million'].iloc[0]
        else:
            oil_share = 0

        st.metric("Oil Sector Share of GDP", f"{oil_share:.1f}%")

    # Column 3: GDP per capita
    with column3:
        # Find GDP per capita data
        gdp_per_capita_data = latest_data[latest_data['Indicators (Current Prices)'].str.contains('GDP per capita', na=False)]

        if not gdp_per_capita_data.empty:
            gdp_per_capita = gdp_per_capita_data['Value \nBD Million'].iloc[0]
        else:
            gdp_per_capita = 0

        st.metric("GDP per Capita (BD)", f"{gdp_per_capita:,.0f}")

    # Column 4: Population
    with column4:
        # Find population data
        population_data = latest_data[latest_data['Indicators (Current Prices)'].str.contains('Population', na=False)]

        if not population_data.empty:
            population = population_data['Value \nBD Million'].iloc[0]
        else:
            population = 0

        st.metric("Population", f"{population:,.0f}")

def main():
    """Main function that creates the dashboard"""

    # Add navigation anchor
    st.markdown('<a name="top"></a>', unsafe_allow_html=True)

    # Create header section with three columns
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])

    with header_col1:
        st.title("Bahrain Economic Dashboard")
        st.markdown("*Comprehensive analysis of Bahrain's economic indicators and Vision 2030 progress*")

    with header_col2:
        if st.button("Quick Stats"):
            st.balloons()
            st.success("Navigate to Key Metrics section below!")

    with header_col3:
        with st.popover("Help"):
            st.markdown("""
            **Quick Guide:**
            - Use **Quick Access** presets for common analysis types
            - **Advanced Filters** for custom year ranges
            - Click sections in the sidebar to focus analysis
            - Use 'Back to Top' button to navigate quickly

            **Analysis Types:**
            - **Overview**: Key metrics and main indicators
            - **Economic**: GDP, diversification, private sector
            - **Employment**: Job market and workforce analysis
            - **Trade**: Import/export and investment flows
            - **Vision 2030**: Future projections and trends
            """)

    # Load all the data
    data = load_data()
    if data is None:
        st.stop()

    # Create sidebar controls
    st.sidebar.header("Dashboard Controls")

    # Quick preset options for different analysis focuses
    st.sidebar.subheader("Quick Access")
    preset = st.sidebar.radio(
        "Choose Analysis Focus:",
        [
            "Overview Dashboard", 
            "Economic Deep Dive", 
            "Employment Focus", 
            "Trade and Investment", 
            "Vision 2030 Analysis", 
            "Custom Selection"
        ],
        index=0
    )

    # Set analysis options based on selected preset
    if preset == "Overview Dashboard":
        analysis_options = ["Key Metrics", "GDP Analysis", "Employment Analysis", "Trade Analysis"]
    elif preset == "Economic Deep Dive":
        analysis_options = ["Key Metrics", "GDP Analysis", "Diversification Metrics", "Private Sector Analysis"]
    elif preset == "Employment Focus":
        analysis_options = ["Key Metrics", "Employment Analysis", "Private Sector Analysis"]
    elif preset == "Trade and Investment":
        analysis_options = ["Key Metrics", "Trade Analysis", "FDI Analysis", "Diversification Metrics"]
    elif preset == "Vision 2030 Analysis":
        analysis_options = ["Key Metrics", "Vision 2030 Time Series"]
    else:  # Custom Selection
        analysis_options = st.sidebar.multiselect(
            "Select Analysis Sections:",
            [
                "Key Metrics", 
                "GDP Analysis", 
                "Employment Analysis", 
                "Trade Analysis", 
                "Diversification Metrics", 
                "Private Sector Analysis", 
                "FDI Analysis", 
                "Vision 2030 Time Series"
            ],
            default=["Key Metrics", "GDP Analysis"]
        )

    # Advanced filters in expandable section
    with st.sidebar.expander("Advanced Filters", expanded=False):
        # Year range selector
        min_year = int(data['economic']['Year'].min())
        max_year = int(data['economic']['Year'].max())
        year_range = st.slider("Year Range:", min_year, max_year, (min_year, max_year))

        # Additional options
        show_projections = st.checkbox("Include 2030 Projections", value=True)
        chart_style = st.selectbox("Chart Style:", ["Modern", "Classic", "Minimal"], index=0)

    # Filter economic data based on selected year range
    filtered_economic = data['economic'][
        (data['economic']['Year'] >= year_range[0]) & 
        (data['economic']['Year'] <= year_range[1])
    ]

    # Display information about selected preset
    if preset != "Custom Selection":
        st.info(f"**{preset}** - Showing {len(analysis_options)} analysis sections")

    # Create progress tracking
    total_sections = len(analysis_options)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Main dashboard content with progress tracking
    section_count = 0

    # Load each selected analysis section
    if "Key Metrics" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: Key Metrics ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_key_metrics_cards(data)
        st.divider()

    if "GDP Analysis" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: GDP Analysis ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_gdp_analysis(filtered_economic)
        st.divider()

    if "Employment Analysis" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: Employment Analysis ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_employment_analysis(data['employment_sector'], data['unemployment_rates'])
        st.divider()

    if "Trade Analysis" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: Trade Analysis ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_trade_analysis(data['trade'])
        st.divider()

    if "Diversification Metrics" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: Diversification Metrics ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_diversification_metrics(filtered_economic, data['trade'])
        st.divider()

    if "Private Sector Analysis" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: Private Sector Analysis ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_private_sector_analysis(filtered_economic)
        st.divider()

    if "FDI Analysis" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: FDI Analysis ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_fdi_analysis(data['inward_fdi'], data['outward_fdi'])
        st.divider()

    if "Vision 2030 Time Series" in analysis_options:
        section_count += 1
        status_text.text(f"Loading: Vision 2030 Time Series ({section_count}/{total_sections})")
        progress_bar.progress(section_count / total_sections)
        create_vision_2030_dashboard(data)
        st.divider()

    # Vision 2030 Goals Assessment
    st.subheader("Vision 2030 Goals Assessment")
    st.markdown("*Based on official government targets and World Bank economic analysis*")

    vision_goals = {
        "Economic Diversification": {
            "target": "Reduce oil dependency to less than 20% of GDP",
            "current_status": "Non-oil sector at 86% of GDP (14% oil dependency) - TARGET ACHIEVED",
            "outlook": "Exceeded target: 14% oil vs 20% target. Key sectors: Financial Services (17.2%), Manufacturing (15.1%), Public Admin (8.5%)"
        },
        "GDP Growth": {
            "target": "Achieve sustainable GDP growth of 3-4% annually",
            "current_status": "3.5% projected for 2025, stabilizing at 3% medium-term",
            "outlook": "On target: Growth driven by BAPCO refinery upgrades completion and Sitra oil refinery expansion"
        },
        "Employment": {
            "target": "Increase private sector employment to 70% of total employment",
            "current_status": "70% employment rate achieved, 1% overall unemployment",
            "outlook": "Target achieved but requires focus on female youth unemployment (12.4%) and labor force participation"
        },
        "Trade Balance": {
            "target": "Maintain positive trade balance through export diversification",
            "current_status": "Current account surplus: 4.8% of GDP (BHD 858.0M), Trade surplus: 7.6% of GDP",
            "outlook": "Strong performance: 9.6% increase in services exports, but vulnerable to energy price volatility"
        },
        "Fiscal Sustainability": {
            "target": "Reduce fiscal deficit and increase non-hydrocarbon revenues",
            "current_status": "Deficit: 8.4% (2024) → 7.7% (2025), despite 64% hydrocarbon revenue reliance",
            "outlook": "Progress with challenges: VAT (10%), corporate tax (15% from Jan 2025), but high debt burden risks"
        },
        "Infrastructure & Digital": {
            "target": "Enhance infrastructure and digital transformation capabilities",
            "current_status": "Economic Recovery Plan (2022-2026) implementation ongoing",
            "outlook": "In progress: Focus on living standards improvement, infrastructure development, and digital transformation"
        }
    }

    for goal, details in vision_goals.items():
        with st.expander(f"{goal}: {details['target']}"):
            st.write(f"**Goal:** {details['target']}")
            st.write(f"**Current Status:** {details['current_status']}")
            st.write(f"**Outlook:** {details['outlook']}")

            # Add progress indicator
            if "On track" in details['outlook'] or "achieved" in details['outlook']:
                st.success("Target achieved or on track")
            elif "Positive" in details['outlook'] or "Improving" in details['outlook']:
                st.info("Positive progress")
            else:
                st.warning("Requires attention")

    # Complete progress tracking
    progress_bar.progress(1.0)
    status_text.text("Dashboard loaded successfully!")

    # Add economic risks and opportunities from official reports
    st.subheader("Economic Risks & Opportunities Assessment")

    risk_col, opp_col = st.columns(2)

    with risk_col:
        st.markdown("""
        **Key Economic Risks (World Bank):**
        - Oil market volatility and global energy demand
        - High interest burden pressure on fiscal balance
        - Potential deficits above 8% of GDP in 2026-27
        - Global growth uncertainty (2.8% in 2025)
        - Continued budget reliance on hydrocarbons (64%)
        """)

    with opp_col:
        st.markdown("""
        **Strategic Opportunities:**
        - BAPCO refinery upgrades completion
        - Sitra oil refinery expansion
        - Digital transformation initiatives
        - Enhanced infrastructure development
        - New corporate tax revenue (15% minimum)
        - Services export growth (9.6% increase)
        """)

    # Add navigation helper
    st.markdown("""
        <style>
        .floating-menu {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
            background: white;
            border-radius: 50px;
            padding: 10px 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 1px solid #ddd;
        }
        </style>
        <div class="floating-menu">
            <a href="#top" style="text-decoration: none; color: #1f77b4;">Back to Top</a>
        </div>
    """, unsafe_allow_html=True)

    # Data sources and methodology information
    with st.expander("Data Sources and Methodology"):
        st.markdown("""
        **Data Sources:**
        - Bahrain Open Data Portal (https://www.data.gov.bh/pages/homepage/)
        - Central Bank of Bahrain
        - Labour Market Regulatory Authority
        - Information and eGovernment Authority

        **Analysis Framework:**
        - GDP growth and sectoral composition analysis
        - Employment patterns by sector and demographics
        - Trade diversification metrics
        - Private sector contribution indicators
        """)

# Run the main function when script is executed
if __name__ == "__main__":
    main()
