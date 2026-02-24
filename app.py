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
    response = requests.get(url)
    res = response.json()
    
    if 'results' in res:
        options = []
        filtered_results = []
        
        for m in res['results']:
            media_type = m.get('media_type')
            if media_type == 'person':
                continue
            
            name = m.get('title') or m.get('name')
            raw_date = m.get('release_date') or m.get('first_air_date')
            
            tag = " [Movie]" if media_type == "movie" else " [TV]"
            date_label = f" ({raw_date[:4]})" if raw_date else ""
            
            options.append(f"{name}{date_label}{tag}")
            filtered_results.append(m)

        if len(filtered_results) > 0:
            if len(filtered_results) == 1:
                selected_item = filtered_results[0]
                st.write(f"Results for: **{options[0]}**")
            else:
                selection = st.selectbox("Which one did you mean?", options[:5])
                idx = options.index(selection)
                selected_item = filtered_results[idx]
            
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
                    st.subheader("Generated movie palette:")
                    for c in final_palette:
                        hex_c = '#%02x%02x%02x' % tuple(c)
                        st.markdown(f'<div style="background-color:{hex_c}; padding:20px; border-radius:10px; margin-bottom:5px; color:white; font-weight:bold; text-shadow: 1px 1px 2px black;">{hex_c}</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    b1, b2 = st.columns(2)
                    
                    if b1.button("💖 Like"):
                        avg_sat = np.mean([max(c)-min(c) for c in final_palette])/255
                        avg_bri = np.mean([sum(c)/3 for c in final_palette])/255
                        st.session_state.pref["sat"] += 0.1 * (avg_sat - st.session_state.pref["sat"])
                        st.session_state.pref["bri"] += 0.1 * (avg_bri - st.session_state.pref["bri"])
                        st.rerun()

                    if b2.button("🗑️ Dislike"):
                        st.session_state.pref["sat"] -= 0.05
                        st.session_state.pref["bri"] -= 0.05
                        st.rerun()
            else:
                st.warning("This item doesn't have a poster available!")
        else:
            st.error("No movies or shows found for that search.")
    else:
        st.error("API Error. Please check your API Key!")

with st.sidebar:
    st.write("Notes: what the AI sees")
    st.progress(min(max(st.session_state.pref['sat'], 0.0), 1.0), text=f"Vibrancy: {st.session_state.pref['sat']:.2f}")
    st.progress(min(max(st.session_state.pref['bri'], 0.0), 1.0), text=f"Brightness: {st.session_state.pref['bri']:.2f}")
    if st.button("Reset AI Brain"):
        st.session_state.pref = {"sat": 0.5, "bri": 0.5}
        st.rerun()
