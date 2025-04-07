# streamlit_app.py
import os
import math
import requests
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.image as mpimg
import streamlit as st

# Streamlit 페이지 설정은 가장 처음에 위치해야 함!
st.set_page_config(
    page_title="기상청 위성 이미지 시각화",
    layout="wide"
)

# 한글 폰트 설정 (나눔고딕 직접 등록)
# customFonts 폴더에 NanumGothic.ttf 파일이 있어야 함
# 현재 파일 기준 상대경로로 customFonts 폴더 안의 NanumGothic.ttf 지정
font_path = os.path.join(os.path.dirname(__file__), 'customFonts', 'NanumGothic.ttf')

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
else:
    import streamlit as st
    st.error("❌ NanumGothic.ttf 파일을 찾을 수 없습니다! 경로를 확인하세요.")

# API 키
my_api_key = 'ONjL08QnRS2Yy9PEJzUtAQ'
api_url = 'https://apihub.kma.go.kr/api/typ01/url/sat_file_down2.php'

# 시간 계산
now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
latest_minute = (now_kst.minute // 2) * 2 - 6

if latest_minute < 0:
    latest_time_kst = now_kst.replace(hour=now_kst.hour - 1, minute=(60 + latest_minute))
else:
    latest_time_kst = now_kst.replace(minute=latest_minute)

latest_time_kst = latest_time_kst.replace(second=0, microsecond=0)
kst_time = latest_time_kst.strftime('%Y%m%d%H%M')

utc_time = datetime.strptime(
    kst_time, '%Y%m%d%H%M'
).replace(
    tzinfo=timezone(timedelta(hours=9))
).astimezone(
    timezone.utc
).strftime('%Y%m%d%H%M')

# 파라미터 설정
data_type = 'img'
level = 'L1B'
area = 'KO'
projection = 'LC'

channels = ['VI004', 'VI005', 'VI006', 'VI008', 'SW038', 'WV063', 'WV073', 'IR133']

params_to_filename = {
    'L1B': 'le1b',
    'VI004': '010',
    'VI005': '010',
    'VI006': '005',
    'VI008': '010',
    'SW038': '020',
    'WV063': '020',
    'WV073': '020',
    'IR133': '020',
}

channel_wavelength = {
    'vi004': '파랑(0.47μm)',
    'vi005': '초록(0.51μm)',
    'vi006': '빨강(0.64μm)',
    'vi008': '식생(0.86μm)',
    'sw038': '적외선(3.8μm)',
    'wv063': '적외선(6.3μm)',
    'wv073': '적외선(7.3μm)',
    'ir133': '적외선(13.3μm)'
}

# 시간 표시 포맷 변경
kst_time_str = latest_time_kst.strftime('%Y년 %m월 %d일 %H시 %M분')

# Streamlit 인터페이스
st.title("🌐 기상청 위성 이미지 다운로드 및 시각화")
st.write(f"**KST 기준 시간:** {kst_time_str}")
progress = st.progress(0)
status_text = st.empty()

file_names = []

# 채널별 다운로드
for idx, channel in enumerate(channels):
    file_name = (
        f'gk2a_ami_{params_to_filename[level]}_{channel.lower()}_'
        f'{area.lower()}{params_to_filename[channel]}{projection.lower()}_{utc_time}.png'
    )
    file_names.append(file_name)

    if not os.path.exists(file_name):
        api_parameters = {
            'typ': data_type,
            'tm': utc_time,
            'lvl': level.lower(),
            'chn': channel.lower(),
            'are': area.lower(),
            'map': projection.lower(),
            'authKey': my_api_key
        }

        response = requests.get(api_url, params=api_parameters)

        if response.status_code != 200 or len(response.content) < 500:
            status_text.warning(f"❌ {channel} 다운로드 실패")
        else:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            status_text.info(f"✅ {file_name} 다운로드 완료")
    else:
        status_text.success(f"📂 {file_name} 이미 존재함")

    progress.progress((idx + 1) / len(channels))

# 시각화
st.subheader("📸 위성 이미지 시각화")

num_images = len(file_names)
num_cols = math.ceil(num_images / 2)

fig, axes = plt.subplots(2, num_cols, figsize=(15, 10))
axes = axes.flatten()

for ax, file_name in zip(axes, file_names):
    img = mpimg.imread(file_name)
    ax.imshow(img)
    ax.axis('off')
    channel = file_name.split('_')[3]
    wavelength = channel_wavelength.get(channel, 'Unknown wavelength')
    ax.set_title(f'{wavelength}')

for i in range(num_images, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
st.pyplot(fig)