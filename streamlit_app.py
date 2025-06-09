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

# Disable file watcher to prevent PyTorch conflicts
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

class ModelManager:
    def __init__(self):
        self._model = None
    
    def get_model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        try:
            # Import PyTorch only when actually needed
            import torch
            from transformers import pipeline
            
            self._model = pipeline(
                "text-generation",
                model="distilgpt2",
                device=0 if torch.cuda.is_available() else -1,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=100
            )
        except ImportError as e:
            st.error(f"Required libraries not installed: {e}")
            self._model = None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self._model = None

# Global model manager
@st.cache_resource
def get_model_manager():
    return ModelManager()

def generate_farming_advice(temp, hum, prec):
    """Generate farming advice based on weather conditions"""
    manager = get_model_manager()
    model = manager.get_model()
    
    if model is None:
        return "Unable to generate AI advice. Please check your environment setup."
    
    prompt = (
        f"Weather Forecast Analysis for Farming:\n"
        f"Temperature: {temp}°C\nHumidity: {hum}%\nPrecipitation: {prec}mm\n\n"
        "Based on these conditions, here are key farming recommendations:\n"
        "1. Crop Selection: With temperature at "
        f"{temp}°C and humidity at {hum}%, suitable crops include"
    )
    
    try:
        result = model(prompt)
        advice = result[0]["generated_text"][len(prompt):].split("\n")[0].strip()
        return advice
    except Exception as e:
        return f"Error generating advice: {str(e)}"

def main():
    st.set_page_config(
        page_title="Farming Advice Generator",
        page_icon="🌾",
        layout="wide"
    )
    
    st.title("🌾 Farming Advice Generator")
    st.markdown("Get AI-powered farming recommendations based on weather conditions")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Weather Parameters")
        temp = st.number_input("Average Temperature (°C)", 0.0, 50.0, 26.7, step=0.1)
        hum = st.number_input("Average Humidity (%)", 0.0, 100.0, 72.4, step=0.1)
        prec = st.number_input("Average Precipitation (mm)", 0.0, 50.0, 4.8, step=0.1)
        
        generate_button = st.button("🚀 Generate Advice", type="primary")
    
    with col2:
        st.subheader("Weather Summary")
        st.metric("Temperature", f"{temp}°C")
        st.metric("Humidity", f"{hum}%")
        st.metric("Precipitation", f"{prec}mm")
    
    if generate_button:
        with st.spinner('Generating personalized farming advice...'):
            advice = generate_farming_advice(temp, hum, prec)
        
        st.subheader("🌱 AI-Generated Farming Advice")
        st.success(advice)
        
        # Add some additional context
        st.info("💡 **Tip**: This advice is generated based on the weather conditions you provided. Always consult with local agricultural experts for region-specific recommendations.")

if __name__ == "__main__":
    main()
