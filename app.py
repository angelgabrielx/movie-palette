import streamlit as st
import requests
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

st.set_page_config(page_title="AI Show Palettes", page_icon="🎭")
st.title("🎭 AI Showstopper Palettes")

if 'pref' not in st.session_state:
    st.session_state.pref = {"sat": 0.5, "bri": 0.5}

API_KEY = "12ccbc23d6be9dc1b2855c9685c441d8"
query = st.text_input("Search for a Musical, Movie, or TV Show:", "Wicked")

if query:
    url = f"https://api.themoviedb.org/3/search/multi?api_key={API_KEY}&query={query}"
    res = requests.get(url).json()
    
    if res['results']:
        options = []
        for m in res['results'][:5]:
            name = m.get('title') or m.get('name')
            date = m.get('release_date') or m.get('first_air_date') or "0000"
            options.append(f"{name} ({date[:4]})")
        
        selection = st.selectbox("Which one did you mean?", options)
        idx = options.index(selection)
        selected_item = res['results'][idx]
        
        if selected_item.get('poster_path'):
            poster = f"https://image.tmdb.org/t/p/w500{selected_item['poster_path']}"
            img = Image.open(requests.get(poster, stream=True).raw).convert('RGB')
            img_array = np.array(img)
            
            pixels = img_array.reshape(-1, 3)
            model = MiniBatchKMeans(n_clusters=12, n_init=3).fit(pixels[::20])
            candidates = model.cluster_centers_.astype(int)
            
            def get_score(c):
                sat = (max(c) - min(c)) / 255
                bri = (sum(c) / 3) / 255
                return abs(sat - st.session_state.pref["sat"]) + abs(bri - st.session_state.pref["bri"])

            final_palette = sorted(candidates, key=get_score)[:5]
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, use_container_width=True)
            
            with col2:
                st.subheader("AI Curated Palette")
                for c in final_palette:
                    hex_c = '#%02x%02x%02x' % tuple(c)
                    st.markdown(f'<div style="background-color:{hex_c}; padding:20px; border-radius:10px; margin-bottom:5px; color:white; font-weight:bold; text-shadow: 1px 1px 2px black;">{hex_c}</div>', unsafe_allow_html=True)
                
                st.write("---")
                b1, b2 = st.columns(2)
                
                if b1.button("💖 Love it"):
                    avg_sat = np.mean([max(c)-min(c) for c in final_palette])/255
                    avg_bri = np.mean([sum(c)/3 for c in final_palette])/255
                    st.session_state.pref["sat"] += 0.1 * (avg_sat - st.session_state.pref["sat"])
                    st.session_state.pref["bri"] += 0.1 * (avg_bri - st.session_state.pref["bri"])
                    st.rerun()

                if b2.button("🗑️ Not for me"):
                    st.session_state.pref["sat"] -= 0.05
                    st.session_state.pref["bri"] -= 0.05
                    st.rerun()
        else:
            st.warning("This item doesn't have a poster available!")

with st.sidebar:
    st.write("### 🤖 Your AI Style Profile")
    st.write(f"Vibrancy Preference: {st.session_state.pref['sat']:.2f}")
    st.write(f"Brightness Preference: {st.session_state.pref['bri']:.2f}")
    if st.button("Reset AI Brain"):
        st.session_state.pref = {"sat": 0.5, "bri": 0.5}
        st.rerun()
