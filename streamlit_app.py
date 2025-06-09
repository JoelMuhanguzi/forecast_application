# import streamlit as st
# import json
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # Cache model loading
# @st.cache_resource
# def load_generation_model(model_name="distilgpt2"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     gen = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device=0 if torch.cuda.is_available() else -1,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         temperature=0.7,
#         num_return_sequences=1
#     )
#     return gen

# @st.cache_resource
# def load_translator(model_name="facebook/nllb-200-distilled-600M", src="eng_Latn", tgt="lug_Latn"):
#     translator = pipeline(
#         "translation",
#         model=model_name,
#         tokenizer=model_name,
#         src_lang=src,
#         tgt_lang=tgt,
#         device=0 if torch.cuda.is_available() else -1
#     )
#     return translator

# # Utility functions from Colab

# def generate_farming_advice(gen, pred):
#     temp, hum, prec = pred['avg_temperature'], pred['avg_humidity'], pred['avg_precipitation']
#     prompt = (
#         f"Weather Forecast Analysis for Farming:\n"
#         f"Temperature: {temp}°C\nHumidity: {hum}%\nPrecipitation: {prec}mm\n\n"
#         f"Based on these conditions, here are key farming recommendations:\n"
#         f"1. Crop Selection: With temperature at {temp}°C and humidity at {hum}%, suitable crops include"
#     )
#     out = gen(prompt, max_new_tokens=100, truncation=True)[0]['generated_text']
#     advice = out[len(prompt):].strip().split("\n")[0]
#     return advice


# def create_structured_advice(pred):
#     temp, hum, prec = pred['avg_temperature'], pred['avg_humidity'], pred['avg_precipitation']
#     advice = {
#         'weather_summary': f"Temperature: {temp}°C, Humidity: {hum}%, Precipitation: {prec}mm",
#         'crop_recommendations': [],
#         'irrigation_advice': '',
#         'pest_management': '',
#         'general_tips': []
#     }
#     # Temperature logic
#     if temp < 15:
#         advice['crop_recommendations'].append('cool-season crops like lettuce, spinach, peas')
#         advice['general_tips'].append('Consider row covers for frost protection')
#     elif temp < 25:
#         advice['crop_recommendations'].append('temperate crops like tomatoes, peppers, beans')
#         advice['general_tips'].append('Ideal temperature for most vegetables')
#     else:
#         advice['crop_recommendations'].append('heat-tolerant crops like okra, sweet potatoes, melons')
#         advice['general_tips'].append('Provide shade during peak hours')
#     # Humidity logic
#     if hum > 80:
#         advice['pest_management'] = 'High humidity increases fungal risk. Ensure good air circulation.'
#         advice['general_tips'].append('Monitor for fungal diseases')
#     elif hum < 40:
#         advice['irrigation_advice'] = 'Low humidity may increase water stress. Monitor soil moisture closely.'
#     else:
#         advice['pest_management'] = 'Moderate humidity levels are favorable for most crops.'
#     # Precipitation logic
#     if prec < 2:
#         advice['irrigation_advice'] = 'Low rainfall expected. Plan for supplemental irrigation.'
#     elif prec > 10:
#         advice['irrigation_advice'] = 'High rainfall expected. Ensure proper drainage.'
#         advice['general_tips'].append('Check drainage systems')
#     else:
#         advice['irrigation_advice'] = 'Moderate rainfall should meet most crop water needs.'
#     return advice

# # Main app

# def main():
#     st.title("🌾 Farming Advice Generator")
#     st.sidebar.header("Input Forecast")
#     temp = st.sidebar.number_input("Average Temperature (°C)", value=26.7, step=0.1)
#     hum = st.sidebar.number_input("Average Humidity (%)", value=72.4, step=0.1)
#     prec = st.sidebar.number_input("Average Precipitation (mm)", value=4.8, step=0.1)
#     tgt_lang = st.sidebar.selectbox("Translate to", ["None", "Swahili (swh_Latn)", "Luganda (lug_Latn)"], index=2)
#     lang_code = tgt_lang.split()[1].strip("()") if tgt_lang != "None" else None

#     if st.button("Generate Advice"):
#         pred = {'avg_temperature': temp, 'avg_humidity': hum, 'avg_precipitation': prec}
#         gen = load_generation_model()
#         advice_text = generate_farming_advice(gen, pred)
#         struct = create_structured_advice(pred)

#         st.subheader("AI-Generated Advice (English)")
#         st.write(advice_text)

#         st.subheader("Structured Recommendations (English)")
#         st.json(struct)

#         if lang_code:
#             translator = load_translator(src="eng_Latn", tgt=lang_code)
#             st.subheader(f"AI-Generated Advice ({lang_code})")
#             translated_text = translator(advice_text)[0]['translation_text']
#             st.write(translated_text)
#             st.subheader(f"Structured Recommendations ({lang_code})")
#             trans_struct = {}
#             for k,v in struct.items():
#                 if isinstance(v, list):
#                     joined = "; ".join(v)
#                     out = translator(joined)[0]['translation_text']
#                     trans_struct[k] = [s.strip() for s in out.split("; ")]
#                 else:
#                     trans_struct[k] = translator(v)[0]['translation_text']
#             st.json(trans_struct)

# if __name__ == "__main__":
#     main()


import streamlit as st
import os
import warnings

# CRITICAL: Set these environment variables BEFORE importing any other libraries
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"

# Suppress warnings
warnings.filterwarnings("ignore")

def load_model_safely():
    """Safely load the model with proper error handling"""
    try:
        # Import torch and transformers only when needed
        import torch
        from transformers import pipeline, logging
        
        # Reduce transformers logging
        logging.set_verbosity_error()
        
        # Load model with better parameters for farming advice
        model = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1,  # Force CPU to avoid GPU issues
            do_sample=True,
            temperature=0.8,
            max_new_tokens=150,
            repetition_penalty=1.2,
            pad_token_id=50256
        )
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

@st.cache_resource
def get_model():
    return load_model_safely()

def generate_better_prompt(temp, hum, prec):
    """Generate a more specific prompt for better farming advice"""
    
    # Determine season/climate conditions
    if temp < 15:
        climate = "cool weather"
    elif temp > 30:
        climate = "hot weather"
    else:
        climate = "moderate weather"
    
    if hum > 80:
        moisture = "high humidity"
    elif hum < 40:
        moisture = "low humidity"  
    else:
        moisture = "moderate humidity"
    
    if prec > 10:
        rainfall = "high rainfall"
    elif prec < 2:
        rainfall = "low rainfall"
    else:
        rainfall = "moderate rainfall"
    
    prompt = f"""Agricultural Advisory Report:
Weather Conditions: {climate}, {moisture}, {rainfall}
Temperature: {temp}°C | Humidity: {hum}% | Precipitation: {prec}mm

Farming Recommendations:
For these conditions, farmers should focus on crops that thrive in {climate} with {moisture}. 
Recommended crops include"""
    
    return prompt

def clean_generated_text(text, max_sentences=3):
    """Clean and format the generated text"""
    # Remove repetitive phrases
    sentences = text.split('.')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:
            # Simple deduplication
            words = set(sentence.lower().split())
            if not any(len(words.intersection(prev_words)) > len(words) * 0.7 for prev_words in seen):
                unique_sentences.append(sentence)
                seen.add(words)
                if len(unique_sentences) >= max_sentences:
                    break
    
    return '. '.join(unique_sentences) + '.' if unique_sentences else text

def provide_fallback_advice(temp, hum, prec):
    """Provide rule-based advice when AI model fails"""
    advice = []
    
    # Temperature-based advice
    if temp < 15:
        advice.append("Cool weather crops: lettuce, spinach, peas, carrots, and cabbage thrive in these temperatures")
    elif temp > 30:
        advice.append("Heat-tolerant crops: tomatoes, peppers, eggplant, okra, and heat-resistant varieties are recommended")
    else:
        advice.append("Moderate temperature crops: most vegetables including beans, corn, squash, and root vegetables will grow well")
    
    # Humidity-based advice  
    if hum > 80:
        advice.append("High humidity requires good air circulation and disease-resistant varieties to prevent fungal issues")
    elif hum < 40:
        advice.append("Low humidity conditions need mulching and frequent watering to retain soil moisture")
    
    # Precipitation-based advice
    if prec > 10:
        advice.append("High rainfall areas should focus on well-draining soils and crops that handle wet conditions")
    elif prec < 2:
        advice.append("Low rainfall requires drought-tolerant crops and efficient irrigation systems")
    
    return " | ".join(advice)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Smart Farming Advisor",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌾 Smart Farming Advisor")
    st.markdown("*AI-powered agricultural recommendations based on weather conditions*")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Weather Parameters")
        
        temp = st.number_input(
            "🌡️ Average Temperature (°C)", 
            min_value=0.0, 
            max_value=50.0, 
            value=26.7, 
            step=0.1,
            help="Enter the average temperature for your area"
        )
        
        hum = st.number_input(
            "💧 Average Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=72.4, 
            step=0.1,
            help="Enter the average humidity percentage"
        )
        
        prec = st.number_input(
            "🌧️ Average Precipitation (mm)", 
            min_value=0.0, 
            max_value=100.0, 
            value=4.8, 
            step=0.1,
            help="Enter the average daily precipitation"
        )
        
        st.divider()
        use_ai = st.checkbox("🤖 Use AI Generation", value=True, help="Uncheck to use rule-based advice")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Current Conditions")
        
        # Weather metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Temperature", f"{temp}°C")
        with metric_col2:
            st.metric("Humidity", f"{hum}%")
        with metric_col3:
            st.metric("Precipitation", f"{prec}mm")
        
        # Climate assessment
        if temp < 15:
            climate_status = "❄️ Cool Climate"
        elif temp > 30:
            climate_status = "🔥 Hot Climate"
        else:
            climate_status = "🌤️ Moderate Climate"
        
        st.info(f"**Climate Assessment:** {climate_status}")
    
    with col2:
        st.subheader("🎯 Generate Advice")
        
        if st.button("🚀 Get Farming Recommendations", type="primary", use_container_width=True):
            
            if use_ai:
                # Try AI generation first
                with st.spinner('🤖 Generating AI-powered advice...'):
                    model = get_model()
                    
                    if model:
                        try:
                            prompt = generate_better_prompt(temp, hum, prec)
                            result = model(prompt, max_length=len(prompt) + 100, num_return_sequences=1)
                            generated = result[0]["generated_text"]
                            advice = generated[len(prompt):].strip()
                            advice = clean_generated_text(advice)
                            
                            if len(advice) < 20 or "crop bites" in advice.lower():
                                # Fallback if AI generates poor output
                                advice = provide_fallback_advice(temp, hum, prec)
                                st.warning("🔄 AI generated unclear advice, using expert rules instead")
                            
                        except Exception as e:
                            advice = provide_fallback_advice(temp, hum, prec)
                            st.warning(f"⚠️ AI model error, using fallback advice: {str(e)}")
                    else:
                        advice = provide_fallback_advice(temp, hum, prec)
                        st.warning("⚠️ AI model unavailable, using rule-based advice")
            else:
                # Use rule-based advice directly
                advice = provide_fallback_advice(temp, hum, prec)
            
            # Display results
            st.subheader("🌱 Farming Recommendations")
            st.success(advice)
            
            # Additional tips
            with st.expander("💡 Additional Tips"):
                st.markdown("""
                **General Farming Tips:**
                - Always test your soil pH before planting
                - Consider crop rotation to maintain soil health
                - Monitor local weather forecasts for sudden changes
                - Consult with local agricultural extension services
                - Keep records of what works best in your specific location
                """)
    
    # Footer
    st.divider()
    st.markdown("*⚠️ This tool provides general guidance. Always consult local agricultural experts for region-specific advice.*")

if __name__ == "__main__":
    main()
