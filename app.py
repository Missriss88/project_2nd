import streamlit as st
from streamlit_folium import st_folium
import folium
import os
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 페이지 설정 ---
st.set_page_config(
    page_title="YOLOv8 & 피부 트러블 분석",
    page_icon="📸",
    layout="wide"
)

# .env 파일에서 KAKAO_API_KEY 값을 찾아 가져옵니다.
KAKAO_MAP_API_KEY = os.getenv("KAKAO_API_KEY")

# API 키가 제대로 로드되었는지 앱 시작 시점에 확인
if not KAKAO_MAP_API_KEY:
    st.error("카카오맵 API 키를 .env 파일에 올바르게 설정해주세요. (KAKAO_API_KEY=your_key)")
    st.stop() # 키가 없으면 앱 실행을 중지

# --- 모델 로드 ---
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

# --- (추가) 클래스 이름 한글 변환 맵 ---
CLASS_NAME_KR_MAP = {
    "Acne": "일반 여드름",
    "Blackhead": "블랙헤드",
    "Cystic": "낭종성 여드름",
    "Flat_wart": "편평 사마귀",
    "Folliculitis": "모낭염",
    "Keloid": "켈로이드",
    "Milium": "비립종",
    "Papular": "구진성 여드름",
    "Purulent": "화농성 여드름",
    "Scars": "흉터",
    "Sebo-crystan-conglo": "피지 낭종성 응괴",
    "Whitehead": "화이트헤드"
}

# --- 화장품 추천 정보 ---
COSMETIC_RECOMMENDATION_MAP = {
    # 비염증성
    "Blackhead": {
        "ingredients": "추천 성분: 살리실산(BHA), 호호바 오일",
        "products": "추천 제품: 클렌징 오일, BHA 토너, 클레이 마스크",
        "keywords": ["BHA", "호호바오일", "클렌징오일"]
    },
    "Whitehead": {
        "ingredients": "추천 성분: AHA, LHA, PHA",
        "products": "추천 제품: 각질 제거 기능의 토너, 필링 패드, 저자극 로션",
        "keywords": ["AHA", "LHA", "PHA"]
    },
    
    # 염증성
    "Papular": {
        "ingredients": "추천 성분: 티트리 오일, 병풀추출물(시카), 나이아신아마이드",
        "products": "추천 제품: 스팟 제품, 시카 성분 진정 세럼/크림",
        "keywords": ["티트리", "시카", "나이아신아마이드"]
    },
    "Purulent": {
        "ingredients": "추천 성분: 살리실산(BHA), 티트리 오일, 어성초 추출물",
        "products": "추천 제품: BHA 스팟 트리트먼트, 어성초 팩",
        "keywords": ["BHA", "티트리", "어성초"]
    },
    
    # 심각한 경우 (설명만 있음)
    "Cystic": { "description": "피부과 전문의 상담 필수. 저자극 보습제(세라마이드, 판테놀) 외 기능성 제품 사용 주의.", "keywords": [] },
    "Folliculitis": { "description": "항균/항진균 관리 필요. 티트리 성분 또는 약산성 클렌저 사용 고려. 전문의 진단 권장.", "keywords": ["티트리"] },
    
    # 흉터 및 자국 관리
    "Scars": { "description": "색소침착: 나이아신아마이드, 비타민C. 패인 흉터: 화장품으로 개선 어려움. 피부과 시술 권장.", "keywords": ["나이아신아마이드", "비타민C"] },
    
    # 나머지 (하나의 설명만 필요한 경우 'description' 키 유지)
    "Keloid": { "description": "화장품으로 관리 불가. 반드시 피부과 전문 치료 필요.", "keywords": [] },
    "Milium": { "description": "화장품으로 제거 불가. 피부과에서 물리적 제거 권장.", "keywords": [] },
    "Flat_wart": { "description": "바이러스성 질환. 화장품으로 치료 불가. 피부과 방문 필수.", "keywords": [] },
    "Acne": { "description": "피부 타입에 맞는 저자극 클렌저와 보습제 사용이 기본입니다.", "keywords": ["여드름"] },
    "Sebo-crystan-conglo": { "description": "복합성 중증. 즉시 피부과 전문의의 집중 치료가 필요합니다.", "keywords": [] }
}

# 모델 파일 경로 지정
DETECTION_MODEL_PATH = r'https://github.com/Missriss88/project_2nd/blob/Missriss88-patch-2/best.pt'

try:
    detection_model = load_yolo_model(DETECTION_MODEL_PATH)
except Exception as e:
    st.error(f"객체 탐지 모델 로딩에 실패했습니다: {e}")
    st.stop()

# --- 분석 및 결과 표시 함수 ---
def analyze_and_display_results(image):
    st.subheader("분석 결과")
    with st.spinner('이미지를 분석 중입니다...'):
        # YOLO 모델로 예측 수행
        results = detection_model.predict(image, verbose=False)
        result = results[0]
        
        # 원본 영어 이름표 (예: {0: 'Acne', 1: 'Blackhead', ...})
        original_names = result.names
        
        # 새로운 한글 이름표를 담을 빈 딕셔너리
        korean_names = {}
        for class_id, eng_name in original_names.items():
            # 위에서 만든 변환표(CLASS_NAME_KR_MAP)를 참고하여 한글 이름을 가져옵니다.
            # 만약 변환표에 없는 이름이면, 원래 영어 이름을 그대로 사용합니다.
            korean_names[class_id] = CLASS_NAME_KR_MAP.get(eng_name, eng_name)
        
        # 모델 결과 객체의 이름표를 통째로 한글 이름표로 교체합니다.
        result.names = korean_names

        # 결과 이미지 표시
        annotated_image_bgr = result.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, caption="탐지 결과", use_column_width=True)

        # OBB 모델의 탐지 결과를 담을 변수 (.obb 속성 확인)
        detections = result.obb 

        if detections is not None and len(detections) > 0:
            st.write("✅ 감지된 피부 트러블 정보:")
            detected_types = {}
            
            # OBB 결과에서 클래스와 신뢰도 추출
            for box in detections:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                confidence = box.conf.item() * 100
                
                # 같은 종류의 트러블 중 가장 확률 높은 것만 저장
                if class_name not in detected_types or confidence > detected_types[class_name]['confidence']:
                    detected_types[class_name] = {'confidence': confidence}

            # 결과 출력
            for acne_type, info in detected_types.items():
                st.markdown(f"### {acne_type} (감지 확률: {info['confidence']:.2f}%)")
                
                # (주의) COSMETIC_RECOMMENDATION_MAP의 키는 여전히 영어이므로,
                # 한글 이름을 다시 영어로 바꾸어 정보를 찾아야 합니다.
                # 이를 위해 영어-한글 역방향 맵을 만듭니다.
                eng_name_map = {v: k for k, v in CLASS_NAME_KR_MAP.items()}
                original_eng_name = eng_name_map.get(acne_type, acne_type)

                recommend_info = COSMETIC_RECOMMENDATION_MAP.get(original_eng_name)
                
                if recommend_info and isinstance(recommend_info, dict):
                    
                    # description, ingredients, products 키에서 값을 가져옴
                    description = recommend_info.get('description')
                    ingredients = recommend_info.get('ingredients')
                    products = recommend_info.get('products')
                    
                    # 표시할 텍스트를 담을 변수
                    display_text = ""

                    if description: # description 키가 있으면 그것만 표시 (심각한 경우 등)
                        display_text = f"{description}"
                    elif ingredients or products: # ingredients나 products가 있으면 조합해서 표시
                        if ingredients:
                            display_text += f"{ingredients}"
                        if products:
                            if display_text: # ingredients가 있었으면 줄바꿈 추가
                                display_text += "<br>"
                            display_text += f"💡 {products}"
                    
                    # 최종적으로 만들어진 텍스트를 st.info로 표시
                    if display_text:
                        # unsafe_allow_html=True를 사용해야 <br> 태그가 작동합니다.
                        st.markdown(f"""
                        <div style="background-color: #e6f3ff; border-left: 5px solid #007bff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            💡 {display_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 검색 키워드 목록(keywords)이 있는지, 비어있지 않은지 확인
                    keywords_list = recommend_info.get('keywords', [])
                    if keywords_list:
                        
                        # 버튼들을 가로로 나열하기 위해 컬럼 생성
                        cols = st.columns(len(keywords_list))
                        
                        # 각 키워드에 대해 버튼 생성
                        for i, keyword in enumerate(keywords_list):
                            with cols[i]:
                                encoded_keyword = requests.utils.quote(keyword)
                                search_url = f"https://www.hwahae.co.kr/search?q={encoded_keyword}&type=products"
                                
                                st.link_button(f"'{keyword}' 화해 랭킹 보기", search_url)
                        
                    st.caption("ⓘ 버튼 클릭시, 해당 성분의 화해 랭킹 페이지로 이동합니다.")
                    st.write("---") # 구분을 위한 라인
                else:
                    st.info("일반적인 피부 관리 권장")

        else:
            st.success("✨ 이미지에서 특별한 피부 트러블이 감지되지 않았습니다.")

# --- 카카오맵 API 키 (반드시 본인의 키로 교체) ---
KAKAO_MAP_API_KEY = os.getenv("KAKAO_API_KEY")

# --- 주소를 좌표로 변환하는 함수 ---
def get_coords_by_keyword(place_name):
    """
    카카오맵 '키워드' 검색 API를 사용해 장소명의 대표 좌표를 가져옵니다.
    (주소 검색 API보다 안정적)
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {"query": place_name, "size": 1} # 가장 정확도가 높은 1개의 결과만 요청
    headers = {"Authorization": f"KakaoAK {KAKAO_MAP_API_KEY}"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if result['documents']:
            first_result = result['documents'][0]
            return first_result['y'], first_result['x'] # 위도(lat), 경도(lng)
        else:
            return None, None
    except Exception as e:
        # 502 오류 등이 발생하면 st.error 대신 None을 반환하여 다음 단계에서 처리
        return None, None

def find_nearby_clinics_kakao(keyword="피부과", lat="37.5665", lng="126.9780"):
    """
    카카오맵 키워드 검색 API를 사용해 주변 장소를 검색합니다.
    (api_key 인자 제거, 전역 변수 KAKAO_MAP_API_KEY 직접 사용)
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {
        "query": keyword, "y": lat, "x": lng, "radius": 3000, "size": 15
    }
    headers = {
        "Authorization": f"KakaoAK {KAKAO_MAP_API_KEY}"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        places = response.json()
        
        if not places.get('documents'):
            return pd.DataFrame()

        results_list = []
        for place in places['documents']:
            results_list.append({
                "이름": place.get('place_name'),
                "주소": place.get('road_address_name') or place.get('address_name'),
                "거리(m)": int(place.get('distance')),
                "카테고리": place.get('category_name'),
                "전화번호": place.get('phone'),
                "지도 바로가기": place.get('place_url'),
                "위도": float(place.get('y')),
                "경도": float(place.get('x'))
            })
        return pd.DataFrame(results_list)
    except Exception as e:
        st.error(f"병원 정보를 가져오는 중 오류 발생: {e}")
        st.info("API 키 또는 네트워크 연결을 확인해주세요.")
        return pd.DataFrame()

# --- 앱 UI 구성 ---
st.title("🩺 피부 트러블 분석 및 솔루션")
st.write("실시간 웹캠 또는 이미지 업로드를 통해 피부 트러블을 분석하고 정보를 얻으세요.")
st.warning("⚠️ 본 분석 결과는 의료적 진단이 아니며, 참고용으로만 활용해 주세요. 정확한 진단은 반드시 전문의와 상담하세요.", icon="❗️")

# --- 탭 생성 ---
tab1, tab2 = st.tabs(["📸 웹캠으로 사진 찍어 분석", "🖼️ 이미지 업로드 분석"])

# --- 웹캠 사진 분석 탭 ---
with tab1:
    st.header("웹캠으로 사진 찍어 분석하기")
    st.info("웹캠을 이용해 피부 사진을 찍으면, AI가 자동으로 분석하여 결과를 보여줍니다.")
    
    picture = st.camera_input("카메라를 보고 피부가 잘 보이도록 한 후, 아래 'Take photo' 버튼을 누르세요.")

    if picture is not None:
        st.subheader("촬영된 사진")
        st.image(picture)
        # 사진을 찍으면 바로 분석 함수 호출
        img = Image.open(picture)
        analyze_and_display_results(img)

# --- 이미지 업로드 분석 탭 ---
with tab2:
    st.header("이미지 파일 업로드하여 분석하기")
    uploaded_file = st.file_uploader(
        "분석할 이미지 파일을 선택하세요...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("업로드된 이미지")
        # 이미지를 업로드하면 바로 분석 함수 호출
        image = Image.open(uploaded_file)
        st.image(image)
        analyze_and_display_results(image)

# --- '내 주변 피부과 찾기' UI 코드 ---
st.markdown("---")
st.subheader("🏥 내 주변 피부과 찾기")

location_query = st.text_input("검색할 지역을 입력하세요 (예: 강남역, 부산 서면)", "서울역")

if st.button("주변 피부과 검색하기 🔍"):
    if not location_query.strip():
        st.warning("검색할 지역을 입력해주세요.")
    else:
        with st.spinner(f"'{location_query}' 주변 피부과를 찾고 있습니다..."):
            
            lat, lng = get_coords_by_keyword(location_query)

            if lat is None or lng is None:
                st.error(f"'{location_query}'의 위치를 찾을 수 없습니다. 더 구체적인 장소나 주소를 입력해보세요.")
            else:
                acne_clinics = find_nearby_clinics_kakao(keyword="여드름 전문 피부과", lat=lat, lng=lng)
                general_clinics = find_nearby_clinics_kakao(keyword="피부과", lat=lat, lng=lng)

                if not acne_clinics.empty or not general_clinics.empty:
                    combined_list = pd.concat([acne_clinics, general_clinics])
                    combined_list = combined_list.drop_duplicates(subset=['이름', '주소'], keep='first').reset_index(drop=True)
                    sorted_list = combined_list.sort_values(by="거리(m)")

                    # st.session_state에 검색 결과와 지도 중심 좌표 저장
                    st.session_state.clinic_list = sorted_list
                    st.session_state.map_center = [lat, lng]
                    st.session_state.location_query = location_query
                else:
                    # 검색 결과가 없을 경우를 대비해 session_state 초기화
                    if 'clinic_list' in st.session_state:
                        del st.session_state.clinic_list
                    st.warning("주변에서 피부과를 찾지 못했습니다.")

# session_state에 병원 목록이 있을 경우에만 지도와 목록을 표시
if 'clinic_list' in st.session_state:
    st.subheader(f"'{st.session_state.location_query}' 주변 지도")
    
    # 지도 생성 (st.session_state 사용)
    m = folium.Map(location=st.session_state.map_center, zoom_start=15)

    # 검색 위치 마커 추가
    folium.Marker(
        st.session_state.map_center,
        popup=folium.Popup(f"<b>{st.session_state.location_query}</b>", max_width=200),
        tooltip="검색 위치",
        icon=folium.Icon(color='blue', icon='star')
    ).add_to(m)

    # 병원 마커 추가
    for idx, row in st.session_state.clinic_list.iterrows():
        # 1. 전화번호에서 하이픈(-) 제거 (더 안정적인 링크 생성을 위해)
        # 전화번호가 없는 경우를 대비해 str()로 감싸줍니다.
        phone_number_link = str(row.get('전화번호', '')).replace('-', '')

        # 2. 팝업에 들어갈 HTML 내용을 만듭니다.
        popup_html = f"""
        <b>{row['이름']}</b><br>
        {row['주소']}<br><br>
        <a href="{row['지도 바로가기']}" target="_blank">🗺️ 카카오맵에서 보기</a>
        """
        
        # 전화번호가 있을 경우에만 전화걸기 링크를 추가합니다.
        if phone_number_link:
            popup_html += f"""
            <br><a href="tel:{phone_number_link}">📞 {row['전화번호']}</a>
            """
        
        # 3. folium.Popup에 HTML 내용을 전달합니다.
        popup = folium.Popup(popup_html, max_width=300)

        # 4. 마커를 생성하고 팝업을 붙입니다.
        folium.Marker(
            location=[row['위도'], row['경도']],
            popup=popup,
            tooltip=row['이름'],
            icon=folium.Icon(color='red', icon='syringe', prefix='fa')
        ).add_to(m)
    
    # Streamlit에 지도 표시
    st_folium(m, width=800, height=600)
    
    # 상세 목록 표시
    st.subheader("상세 목록")
    st.write(f"총 **{len(st.session_state.clinic_list)}개**의 주변 피부과를 찾았습니다.")
    
    st.dataframe(
        st.session_state.clinic_list,
        column_config={
            "지도 바로가기": st.column_config.LinkColumn("Kakao Map 링크", display_text="바로가기 🗺️"),
            "위도": None,
            "경도": None
        },
        hide_index=True
    )