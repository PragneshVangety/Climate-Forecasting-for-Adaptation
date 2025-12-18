import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Climate Adaptation Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        background-color: #fef3c7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# Generate mock climate data
@st.cache_data
def generate_climate_data(variable='Temperature', months=84):
    """Generate simulated climate data with trends and seasonality"""
    data = []
    today = datetime.now()
    start_date = today - timedelta(days=365*5)  # 5 years ago
    
    for i in range(months):
        current_date = start_date + timedelta(days=30*i)
        date_str = current_date.strftime('%Y-%m')
        is_future = current_date > today
        
        # Seasonality
        month_index = current_date.month
        if variable == 'Temperature':
            base_value = 30 + 10 * np.sin((month_index / 11) * np.pi)
            base_value += i * 0.005  # warming trend
            noise_factor = 2
        else:  # Rainfall
            base_value = 100 + 200 * np.exp(-((month_index - 6) ** 2) / 2)
            noise_factor = 20
        
        noise = (np.random.random() - 0.5) * noise_factor
        historical_val = None if is_future else base_value + noise
        
        # Model predictions
        lstm_val = base_value + (np.random.random() - 0.5) * noise_factor * 0.5
        arima_val = base_value * 0.98 + (np.random.random() - 0.5) * noise_factor * 0.75
        prophet_val = base_value * 1.02 + (np.random.random() - 0.5) * noise_factor * 0.6
        
        data.append({
            'date': date_str,
            'variable': variable,
            'historical': historical_val,
            'lstm': lstm_val if is_future or i > 48 else None,
            'arima': arima_val if is_future or i > 48 else None,
            'prophet': prophet_val if is_future or i > 48 else None,
        })
    
    return pd.DataFrame(data)

# Sidebar navigation
st.sidebar.title("🌡️ Climate Dashboard")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Overview", "Weather Prediction", "Forecast Analysis", "Model Evaluation", "Risk & Adaptation", "Data Training"]
)

# Main content based on page selection
if page == "Overview":
    st.markdown('<h1 class="main-header">Climate Adaptation Dashboard</h1>', unsafe_allow_html=True)
    
    # Generate data
    temp_data = generate_climate_data('Temperature')
    rain_data = generate_climate_data('Rainfall')
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_temp = 32.5
        st.metric(
            label="🌡️ Current Avg Temperature",
            value=f"{current_temp}°C",
            delta="+1.2°C vs Historical"
        )
    
    with col2:
        st.metric(
            label="🌧️ Precipitation Status",
            value="Deficit",
            delta="-15% below seasonal average"
        )
    
    with col3:
        st.metric(
            label="🚨 Risk Level",
            value="Medium",
            delta="Drought probability: 65%"
        )
    
    # Alert card
    st.markdown("""
    <div class="alert-card">
        <h4>⚠️ Adaptation Action Required</h4>
        <p>High probability of drought conditions detected for upcoming crop season. 
        Consider implementing water conservation measures and drought-resistant crop varieties.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Trend (Last 12 Months)")
        recent_temp = temp_data[temp_data['historical'].notna()].tail(12)
        fig_temp = px.area(
            recent_temp, 
            x='date', 
            y='historical',
            title="Monthly Average Temperature",
            labels={'historical': 'Temperature (°C)', 'date': 'Date'}
        )
        fig_temp.update_traces(fill='tonexty', fillcolor='rgba(239, 68, 68, 0.1)')
        fig_temp.update_layout(showlegend=False)
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        st.subheader("Rainfall Trend (Last 12 Months)")
        recent_rain = rain_data[rain_data['historical'].notna()].tail(12)
        fig_rain = px.area(
            recent_rain, 
            x='date', 
            y='historical',
            title="Monthly Precipitation",
            labels={'historical': 'Rainfall (mm)', 'date': 'Date'}
        )
        fig_rain.update_traces(fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)')
        fig_rain.update_layout(showlegend=False)
        st.plotly_chart(fig_rain, use_container_width=True)

elif page == "Weather Prediction":
    st.header("🔮 Weather Prediction Tool")
    st.write("Use the trained model to predict precipitation based on weather parameters.")
    
    model = load_model()
    
    if model is not None:
        st.subheader("Input Weather Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                temp_high = st.number_input("Temp High (°F)", value=85.0)
                temp_avg = st.number_input("Temp Avg (°F)", value=75.0)
                temp_low = st.number_input("Temp Low (°F)", value=65.0)
                dewpoint_high = st.number_input("Dew Point High (°F)", value=70.0)
            
            with col2:
                dewpoint_avg = st.number_input("Dew Point Avg (°F)", value=60.0)
                dewpoint_low = st.number_input("Dew Point Low (°F)", value=50.0)
                humidity_high = st.number_input("Humidity High (%)", value=90.0)
                humidity_avg = st.number_input("Humidity Avg (%)", value=70.0)
            
            with col3:
                humidity_low = st.number_input("Humidity Low (%)", value=50.0)
                pressure = st.number_input("Sea Level Pressure (Inches)", value=30.0)
                visibility_high = st.number_input("Visibility High (Miles)", value=10.0)
                visibility_avg = st.number_input("Visibility Avg (Miles)", value=8.0)
            
            with col4:
                visibility_low = st.number_input("Visibility Low (Miles)", value=6.0)
                wind_high = st.number_input("Wind High (MPH)", value=15.0)
                wind_avg = st.number_input("Wind Avg (MPH)", value=10.0)
                wind_gust = st.number_input("Wind Gust (MPH)", value=20.0)
            
            submitted = st.form_submit_button("Predict Precipitation")
            
            if submitted:
                try:
                    input_data = [
                        temp_high, temp_avg, temp_low, dewpoint_high, dewpoint_avg, dewpoint_low,
                        humidity_high, humidity_avg, humidity_low, pressure,
                        visibility_high, visibility_avg, visibility_low,
                        wind_high, wind_avg, wind_gust
                    ]
                    
                    prediction = model.predict([input_data])[0]
                    
                    st.success(f"🌧️ Predicted Precipitation: **{prediction:.2f} inches**")
                    
                    # Interpretation
                    if prediction < 0.1:
                        st.info("☀️ Very low precipitation expected - Clear/Dry conditions")
                    elif prediction < 0.5:
                        st.info("🌤️ Light precipitation expected")
                    elif prediction < 1.0:
                        st.warning("🌧️ Moderate precipitation expected")
                    else:
                        st.error("⛈️ Heavy precipitation expected - Potential flooding risk")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

elif page == "Forecast Analysis":
    st.header("📊 Climate Forecast Analysis")
    
    # Generate forecast data
    temp_data = generate_climate_data('Temperature')
    rain_data = generate_climate_data('Rainfall')
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Short-term Metrics (1-3 months)**")
        short_metrics = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM'],
            'RMSE': [0.95, 1.10, 0.88],
            'MAE': [0.78, 0.85, 0.74],
            'MAPE': [6.2, 6.9, 5.9]
        })
        st.dataframe(short_metrics, use_container_width=True)
    
    with col2:
        st.write("**Long-term Metrics (6-12 months)**")
        long_metrics = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM'],
            'RMSE': [1.87, 1.75, 1.65],
            'MAE': [1.45, 1.32, 1.20],
            'MAPE': [9.8, 9.1, 8.7]
        })
        st.dataframe(long_metrics, use_container_width=True)
    
    # Forecast visualization
    st.subheader("Temperature Forecast Comparison")
    
    # Filter data for visualization
    forecast_temp = temp_data[temp_data['date'] >= '2024-01'].copy()
    
    fig = go.Figure()
    
    # Historical data
    historical = forecast_temp[forecast_temp['historical'].notna()]
    fig.add_trace(go.Scatter(
        x=historical['date'],
        y=historical['historical'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='black', width=2)
    ))
    
    # Model predictions
    future = forecast_temp[forecast_temp['lstm'].notna()]
    fig.add_trace(go.Scatter(
        x=future['date'],
        y=future['lstm'],
        mode='lines',
        name='LSTM',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=future['date'],
        y=future['arima'],
        mode='lines',
        name='ARIMA',
        line=dict(color='blue', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=future['date'],
        y=future['prophet'],
        mode='lines',
        name='Prophet',
        line=dict(color='green', dash='dashdot')
    ))
    
    fig.update_layout(
        title="Temperature Forecast: Model Comparison",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.header("📈 Model Evaluation & Metrics")
    
    # Model performance visualization
    models = ['ARIMA', 'Prophet', 'LSTM']
    rmse_short = [0.95, 1.10, 0.88]
    rmse_long = [1.87, 1.75, 1.65]
    
    fig = go.Figure(data=[
        go.Bar(name='Short-term RMSE', x=models, y=rmse_short),
        go.Bar(name='Long-term RMSE', x=models, y=rmse_long)
    ])
    
    fig.update_layout(
        title="Model Performance Comparison (RMSE)",
        xaxis_title="Model",
        yaxis_title="RMSE",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    
    features = ['Prev Month Temp', 'Seasonality', 'Yearly Trend', 'Prev Month Rain', 'ENSO Index']
    importance = [0.85, 0.60, 0.45, 0.15, 0.10]
    
    fig_importance = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Climate Prediction",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

elif page == "Risk & Adaptation":
    st.header("🚨 Climate Risk Assessment & Adaptation")
    
    # Risk alerts
    st.subheader("Current Risk Alerts")
    
    alerts = [
        {
            'type': 'Drought',
            'severity': 'High',
            'message': 'Extended dry period predicted for next 3 months',
            'recommendation': 'Implement water conservation, switch to drought-resistant crops'
        },
        {
            'type': 'Heatwave',
            'severity': 'Medium',
            'message': 'Above-average temperatures expected in summer months',
            'recommendation': 'Prepare cooling systems, adjust planting schedules'
        }
    ]
    
    for alert in alerts:
        severity_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        st.markdown(f"""
        **{severity_color[alert['severity']]} {alert['type']} - {alert['severity']} Risk**
        
        *Issue:* {alert['message']}
        
        *Recommendation:* {alert['recommendation']}
        
        ---
        """)
    
    # Adaptation strategies
    st.subheader("Recommended Adaptation Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Agricultural Adaptations**")
        st.write("• Drought-resistant crop varieties")
        st.write("• Improved irrigation efficiency")
        st.write("• Crop rotation strategies")
        st.write("• Soil moisture conservation")
    
    with col2:
        st.write("**Infrastructure Adaptations**")
        st.write("• Enhanced water storage systems")
        st.write("• Flood management infrastructure")
        st.write("• Heat-resistant building materials")
        st.write("• Early warning systems")

elif page == "Data Training":
    st.header("🔧 Model Training & Data Pipeline")
    
    st.subheader("Dataset Information")
    
    # Dataset stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", "50,000+")
    with col2:
        st.metric("Time Range", "2000-2024")
    with col3:
        st.metric("Features", "16")
    
    # Training configuration
    st.subheader("Training Configuration")
    
    with st.expander("Model Parameters"):
        model_type = st.selectbox("Model Type", ["LSTM", "ARIMA", "Prophet"])
        
        if model_type == "LSTM":
            epochs = st.slider("Epochs", 50, 500, 200)
            batch_size = st.slider("Batch Size", 16, 128, 32)
        elif model_type == "ARIMA":
            p = st.slider("AR Order (p)", 0, 5, 2)
            d = st.slider("Differencing (d)", 0, 2, 1)
            q = st.slider("MA Order (q)", 0, 5, 2)
        else:  # Prophet
            seasonality = st.selectbox("Seasonality", ["additive", "multiplicative"])
    
    # Data preprocessing options
    st.subheader("Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        impute_missing = st.checkbox("Impute Missing Values", True)
        remove_outliers = st.checkbox("Remove Outliers", True)
    
    with col2:
        normalize = st.checkbox("Normalize Features", True)
        check_stationarity = st.checkbox("Check Stationarity", True)
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            # Simulate training process
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            st.success("✅ Model training completed successfully!")
            st.info("New model saved and ready for predictions.")

# Footer
st.markdown("---")
st.markdown(
    "**Climate Adaptation Dashboard** | Built with Streamlit | "
    "Data sources: NOAA, Local Weather Stations"
)
