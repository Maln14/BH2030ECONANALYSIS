Bahrain Economic Dashboard & Vision 2030 Analysis
A comprehensive, interactive economic analysis dashboard for Bahrain featuring advanced time series forecasting, economic diversification metrics, and Vision 2030 progress tracking. Built with Streamlit and deployed on Streamlit.

 Project Overview
This project provides a sophisticated analytical framework for understanding Bahrain's economic performance through multiple lenses:

Historical economic trend analysis:
Advanced time series forecasting for Vision 2030 projections, Economic diversification progress tracking, Employment and trade pattern analysis, Foreign Direct Investment (FDI) flow analysis.

Live Dashboard:
Access the live dashboard: 

Architecture & Implementation:
Core Components:
main.py - Primary dashboard application with modular analysis functions
BH2030Analysis.py - Advanced time series analysis and forecasting module
consolidated_data.py - Data consolidation and preprocessing utilities

Technology Stack:
Frontend: Streamlit (Interactive web application framework), Data Processing: Pandas, NumPy, Visualization: Plotly (Interactive charts), Matplotlib, Statistical Analysis: Scikit-learn, Statsmodels, SciPy, Deployment: Streamlit.

Methodology & Logic:
1. Data Architecture & Processing Pipeline
Data Sources Integration:

# Nine primary datasets consolidated:
data_files = 
    'BHRGenEconIndic.csv',          # GDP, sector contributions, 
    'BHEmploybySec.csv',            # Employment by sector/gender,
    'BHEmployRates.csv',            # Employment participation rates,
    'BHTradeStats.csv',             # Import/export statistics,
    'BHUnempAge&Gender.csv',        # Unemployment demographics,
    'BHUnemployRates.csv',          # Unemployment trends,
    'ECONIndicAgricultral.csv',     # Agricultural indicators,
    'InwardFDI.csv',                # Foreign investment inflows,
    'OutwardFDI.csv'                # Bahraini investment abroad.

Data Quality & Validation Process:
Load Phase: CSV files loaded with error handling and type validation, Clean Phase: Missing values identified and handled appropriately, Transform Phase: Standardized column naming and data format conversion, Validate Phase: Data consistency checks and outlier detection, Cache Phase: Streamlit caching for optimal performance.
#Note: Primariy source of data, https://www.data.gov.bh/pages/homepage/ , is limited by rather narrow accumulation of data, many datasets were recently consolidated (2020's and onwards), while other datasets only go as far back as 2016 and a few older to the 2000's. Thus, this analysis is capped by the quanitity of data provided. 

2. Economic Analysis Framework
GDP Analysis Logic:
    # Multi-dimensional GDP analysis approach:
    # 1. Growth rate visualization over time, # 2. Sectoral composition (Oil vs Non-oil), # 3. Comparative analysis with Vision 2030 targets.

Analytical Approach:

Temporal Analysis: Tracks GDP growth patterns over multiple years, Sectoral Decomposition: Separates oil-dependent vs diversified economic activity, Comparative Benchmarking: Measures progress against Vision 2030 targets, Economic Diversification Metrics.

Diversification Index Calculation:

# Non-oil share calculation methodology
nonoil_share = (
    (merged_data['Value (Million BD)_total'] - merged_data['Value (Million BD)_oil']) / 
    merged_data['Value (Million BD)_total'] * 100
)
Vision 2030 Target Tracking:

Target: Reduce oil dependency to <20% of GDP
Current Status: 14% oil dependency (TARGET ACHIEVED)
Methodology: Continuous monitoring with visual threshold indicators

3. Advanced Time Series Analysis & Forecasting
Multi-Model Forecasting Approach
The BahrainTimeSeriesAnalysis class implements three complementary forecasting methodologies:

1. Linear Regression Projection
def linear_projection(self, ts_data):
    X = ts_data['Year'].values.reshape(-1, 1)
    y = ts_data['Value'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate 2024-2030 projections
    future_X = self.projection_years.reshape(-1, 1)
    future_pred = model.predict(future_X)
    
    return future_pred, r2_score, mean_absolute_error

2. Polynomial Regression (Degree 2)
def polynomial_projection(self, ts_data, degree=2):
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

   # Captures non-linear growth patterns
4. Exponential Smoothing
def exponential_smoothing_projection(self, ts_data):
    # Multiple model comparison:
    # - Simple exponential smoothing
    # - Linear trend exponential smoothing
    # - Best model selection based on AIC criteria

Trend Analysis Algorithm
Multi-Dimensional Trend Identification:

def identify_trend(self, ts_data, indicator_name):
    # 1. Linear trend calculation using regression slope
    # 2. Trend strength assessment via R-squared
    # 3. Volatility measurement (coefficient of variation)
    # 4. Cyclical pattern detection (peaks/valleys analysis)
    # 5. Direction classification (Increasing/Decreasing/Stable)
Statistical Metrics Generated:

Trend Direction: Increasing/Decreasing/Stable based on slope
Trend Strength: Strong/Moderate/Weak based on R²
Volatility: Standard deviation relative to mean
Cyclical Patterns: Peak/valley frequency analysis
Total Change: Percentage change from start to end period

4. Trade Analysis & Export Diversification
Stacked Area Visualization Logic
# Advanced export composition analysis
export_chart.add_trace(go.Scatter(
    x=merged_exports['Year'],
    y=merged_exports['Value (Million BD)_nonoil'],
    fill='tozeroy',  # Fill to zero baseline
    name='Non-Oil Exports'
))
export_chart.add_trace(go.Scatter(
    x=merged_exports['Year'],
    y=merged_exports['Value (Million BD)_oil'] + merged_exports['Value (Million BD)_nonoil'],
    fill='tonexty',  # Stack on previous layer
    name='Oil Exports'
))
Export Diversification Assessment:

Baseline Measurement: 50% non-oil exports as balanced portfolio target
Trend Analysis: Multi-year progression toward diversification
Visual Indicators: Color-coded progress tracking

5. Employment & Labor Market Analysis
Multi-Dimensional Employment Framework
Sectoral Employment Distribution: Gender-disaggregated analysis by economic sector
Unemployment Trend Analysis: Time series of unemployment rates by demographic
Labor Force Participation: Comprehensive workforce engagement metrics
Analytical Logic:

# Gender-sector employment analysis
sector_summary = employment_data.groupby(['Sector', 'Sex'])['Recruitment'].sum().reset_index()
# Unemployment trend visualization
unemployment_rates = unemployment_data[unemployment_data['Indicator'].str.contains('Rate', na=False)]
6. Foreign Direct Investment (FDI) Analysis
Comprehensive FDI Framework
Inward vs Outward FDI Balance:

# Net FDI position calculation
fdi_balance['Net_FDI'] = fdi_balance['Inward_FDI'] - fdi_balance['Outward_FDI']
# Sector-wise FDI trend analysis for top 5 performing sectors
top_sectors = inward_fdi[inward_fdi['Year'] == latest_year].nlargest(5, 'Value.')
Investment Flow Analysis:

Sectoral Performance: Top 5 FDI recipient sectors over time
Net Position Tracking: Balance between inward and outward investment
Trend Identification: Growth patterns and volatility assessment
7. Vision 2030 Goals Assessment Framework
Multi-Goal Progress Tracking System
Six Core Vision 2030 Goals Monitored:

Economic Diversification: <20% oil dependency ( ACHIEVED: 14%)
GDP Growth: 3-4% sustainable growth ( ON TRACK: 3.5% projected 2025)
Employment: 70% private sector employment ( ACHIEVED)
Trade Balance: Positive balance maintenance ( ACHIEVED: 4.8% surplus)
Fiscal Sustainability: Deficit reduction & non-hydrocarbon revenue increase
Infrastructure & Digital: Digital transformation capabilities enhancement
Progress Indicator Logic:

# Automated progress assessment
if "achieved" in details['outlook']:
    st.success("Target achieved or on track")
elif "Positive" in details['outlook']:
    st.info("Positive progress")
else:
    st.warning("Requires attention")
 Implementation Steps & Development Process
Phase 1: Data Collection & Consolidation
Data Source Identification: Nine official Bahrain government datasets
Data Quality Assessment: Missing value analysis and consistency checks
Consolidation Script Development: consolidated_data.py for unified data processing
Validation Framework: Error handling and data integrity verification

Phase 2: Core Dashboard Development
Streamlit Application Structure: Modular function-based architecture
Interactive Interface Design: Sidebar controls with preset configurations
Visualization Implementation: Plotly-based interactive charts
Performance Optimization: Caching strategies and efficient data loading

Phase 3: Advanced Analytics Integration
Time Series Analysis Module: BH2030Analysis.py development
Forecasting Model Implementation: Multi-model approach for robust projections
Statistical Validation: R-squared, MAE, and trend strength calculations
Vision 2030 Progress Tracking: Automated goal assessment framework

Phase 4: User Experience Enhancement
Navigation System: Quick access presets and custom selection options
Progress Tracking: Real-time loading indicators and status updates
Help System: Integrated documentation and user guidance
Responsive Design: Multi-device compatibility

Phase 5: Deployment & Documentation
Configuration: Streamlit server setup for 0.0.0.0:5000
Workflow Integration: Run button configuration for easy deployment
Documentation Creation: Comprehensive README and inline documentation
User Testing: Interface usability and analytical accuracy verification

Dashboard Features & Analysis Sections
Quick Access Presets
Overview Dashboard: Key metrics + GDP + Employment + Trade
Economic Deep Dive: GDP + Diversification + Private Sector
Employment Focus: Labor market analysis + Private sector employment
Trade and Investment: Trade patterns + FDI + Diversification
Vision 2030 Analysis: Advanced forecasting + Goal tracking
Custom Selection: User-defined analysis combination
Key Performance Indicators (KPIs)

Real-time Metrics Dashboard:

GDP (Million BD)
Oil Sector Share of GDP (%)
GDP per Capita (BD)
Population Count
Advanced Analytics Features
Multi-Model Forecasting: Linear, Polynomial, Exponential Smoothing
Trend Analysis: Direction, Strength, Volatility, Cyclical Patterns
Comparative Analysis: Multi-indicator visualization
Interactive Filtering: Year range selection and chart style options

Visualization Design Philosophy:
Chart Selection Rationale
Time Series Data: Line charts with markers for trend clarity
Compositional Data: Stacked area charts for part-to-whole relationships
Comparative Data: Grouped bar charts for category comparison
Performance Metrics: Metric cards with delta indicators
Forecasting: Multi-line charts with differentiated projection styles
Color Coding Strategy
Green: Positive indicators, non-oil sectors, achieved targets
Blue: Historical data, primary metrics
Red/Orange: Oil sectors, warning indicators, projection lines
Gray: Reference lines, neutral indicators

Technical Configuration:
Streamlit Configuration
st.set_page_config(
    page_title="BH Economic Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

Data Caching Strategy:
@st.cache_data
def load_data():
    # Optimized data loading with error handling
    # Prevents repeated CSV reading on user interactions

Deployment Configuration
Server Address: 0.0.0.0 (accessible to external users)
Workflow: Streamlit server with production-ready settings

Usage Instructions:
For Analysts & Researchers
Quick Start: Select "Overview Dashboard" preset for comprehensive view
Deep Analysis: Use "Economic Deep Dive" for detailed economic metrics
Custom Research: Choose "Custom Selection" for specific indicator combinations
Forecasting: Select "Vision 2030 Analysis" for advanced projections
For Policymakers
Goal Tracking: Monitor Vision 2030 progress in dedicated section
Risk Assessment: Review Economic Risks & Opportunities section
Comparative Analysis: Use year range filters for policy period assessment
Export Reports: Download charts and data for presentations

For Students & Citizens
Educational Mode: Start with "Overview Dashboard" for basic understanding
Interactive Learning: Hover over charts for detailed information
Help System: Use the help popover for guidance
Progressive Exploration: Move from overview to specialized analysis sections

Data Quality & Limitations:
Data Sources Reliability
Official Sources: All data from Bahrain government agencies
Update Frequency: Varies by indicator (annual for most economic data)
Coverage Period: 2000-2023 for most indicators
Missing Data: Handled with appropriate interpolation or exclusion

Analytical Limitations:
Forecasting Accuracy: Models based on historical trends; external shocks not predicted
Data Lag: Some indicators have 1-2 year reporting delays
Model Assumptions: Linear and polynomial projections assume trend continuation

Future Enhancements:
Contemplating a couple of feautures including: Real-time Data Integration,Comparative Analysis with GCC Countries, Scenario Modeling, Export Functionality,and Mobile Optimization.

References & Data Sources:
Primary Data Sources:
Bahrain Open Data Portal
Central Bank of Bahrain
Labour Market Regulatory Authority
Information and eGovernment Authority
Bahrain Economic Development Board

Analytical References
World Bank Bahrain Economic Outlook Reports
Bahrain Vision 2030 Official Documents
Central Bank of Bahrain Annual Reports
International Monetary Fund Country Reports

Contributing:
This project is open for contributions. Areas for improvement:

Additional economic indicators
Enhanced forecasting models
User interface improvements
Documentation enhancements

Contact & Support:
For technical issues, feature requests, or analytical questions:

GitHub Issues: 
Documentation: Refer to inline help system
Live Demo:
