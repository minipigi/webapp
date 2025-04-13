# -*- coding: utf-8 -*-
"""
천체 관측 가능지수 측량기 (COI Measurement Tool)
- 위치 지정, 위성 이미지, 태양/달 고도 그래프, 기상 정보, 천체 관측 가능지수를 제공하는 Streamlit 애플리케이션
"""

# 필요한 라이브러리 임포트
import os
import io
import base64
import time
from datetime import datetime, timedelta, timezone

# 데이터 처리 및 변환 관련 라이브러리
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from pyproj import Transformer

# 이미지 처리 라이브러리
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 웹 요청 및 파싱 관련 라이브러리
import requests
import ssl
from requests.adapters import HTTPAdapter

# 천문학 관련 라이브러리
from skyfield.api import load, Topos

# Streamlit 관련 라이브러리
import streamlit as st
import pytz

# ===== 상수 및 설정 =====
API_KEY = 'ONjL08QnRS2Yy9PEJzUtAQ'

# ===== 유틸리티 함수 =====
def setup_korean_font():
    """한글 폰트 설정"""
    font_path = os.path.join(os.path.dirname(__file__), 'customFonts', 'NanumGothic.ttf')
    
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        return True
    else:
        st.error("❌ NanumGothic.ttf 파일을 찾을 수 없습니다! 경로를 확인하세요.")
        return False


def image_to_base64(image_path):
    """이미지를 Base64로 인코딩"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    except Exception as e:
        st.error(f"이미지 인코딩 실패: {e}")
        return None


def render_img_html(image_b64):
    """Base64 인코딩 이미지를 HTML로 렌더링"""
    st.markdown(
        f"<img style='max-width: 100%;max-height: 100%;' src='data:image/jpeg;base64, {image_b64}'/>", 
        unsafe_allow_html=True
    )


# ===== 기상 관련 클래스 및 함수 =====
class TLSAdapter(HTTPAdapter):
    """TLS 보안 설정을 위한 어댑터"""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)


def convert_to_tm(lat, lon):
    """위도/경도를 TM 중부원점 좌표로 변환"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    return transformer.transform(lon, lat)


def get_nearest_station(tmX, tmY):
    """가장 가까운 대기질 측정소 이름 얻기"""
    session = requests.Session()
    session.mount('https://', TLSAdapter())

    url = "https://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getNearbyMsrstnList"
    params = {
        "serviceKey": "CIixk5nqh86hsMOFj1C3YPp4LxndXeRB848pZgiKuSmSbbLRBgfZqAReCjaDxDdfi2q8GW5N1Z0+ilyWpN4Epg==",
        "returnType": "xml",
        "tmX": tmX,
        "tmY": tmY,
        "ver": "1.1"
    }

    response = session.get(url, params=params)
    tree = ET.fromstring(response.content)
    station_name = tree.findtext(".//item/stationName")
    return station_name


def get_air_quality(station_name):
    """대기질 정보 얻기"""
    session = requests.Session()
    session.mount('https://', TLSAdapter())

    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        "serviceKey": "CIixk5nqh86hsMOFj1C3YPp4LxndXeRB848pZgiKuSmSbbLRBgfZqAReCjaDxDdfi2q8GW5N1Z0+ilyWpN4Epg==",
        "returnType": "xml",
        "numOfRows": "1",
        "pageNo": "1",
        "stationName": station_name,
        "dataTerm": "DAILY",
        "ver": "1.0"
    }

    response = session.get(url, params=params)
    tree = ET.fromstring(response.content)
    item = tree.find(".//item")

    if item is not None:
        pm10 = item.findtext("pm10Value")
        pm25 = item.findtext("pm25Value")
        o3 = item.findtext("o3Value")
        return pm10, pm25, o3
    return None, None, None


def get_base_time(service):
    """서비스 유형에 따른 기상 정보 기준 시간 계산"""
    now = datetime.now(timezone.utc) + timedelta(hours=9)  # UTC를 KST로 변환
    hour = now.hour  # KST 기준의 시간
    minute = now.minute  # KST 기준의 분

    if service == '초단기실황':
        # 정시 데이터는 정시에서 10분 이후에 제공 가능
        if minute < 10:
            hour -= 1
            if hour < 0:
                now -= timedelta(days=1)
                hour = 23
        return now.strftime("%Y%m%d"), f"{hour:02}00"

    elif service == '초단기예보':
        if minute < 40:
            hour -= 1
            if hour < 0:
                now -= timedelta(days=1)
                hour = 23
        return now.strftime("%Y%m%d"), f"{hour:02}30"

    elif service == '단기예보':
        # 발표 가능한 기준 시간 목록
        times = [2, 5, 8, 11, 14, 17, 20, 23]

        # 현재 시각 기준으로 가장 가까운 base_time 구하기
        for t in reversed(times):
            base_dt = now.replace(hour=t, minute=10, second=0, microsecond=0)
            if now >= base_dt:
                base_hour = t
                break
        else:
            # 새벽 00~02:10 이전엔 전날 23시 예보를 기준
            base_hour = 23
            now -= timedelta(days=1)

        return now.strftime("%Y%m%d"), f"{base_hour:02}00"


def call_weather_api(service_name, api_url, nx, ny):
    """기상청 API 호출"""
    base_date, base_time = get_base_time(service_name)
    params = {
        'pageNo': 1,
        'numOfRows': 1000,
        'dataType': 'XML',
        'base_date': base_date,
        'base_time': base_time,
        'nx': nx,
        'ny': ny,
        'authKey': API_KEY
    }
    response = requests.get(api_url, params=params)
    return base_date, base_time, response


# ===== 위성 이미지 관련 함수 =====
def download_satellite_images():
    """기상청 위성 이미지 다운로드"""
    # API 키 및 URL 설정
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
                'authKey': API_KEY
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

    return file_names, channel_wavelength


def combine_satellite_images(file_names, channel_wavelength):
    """위성 이미지 합치기"""
    # 폰트 설정
    font_path = os.path.join(os.path.dirname(__file__), 'customFonts', 'NanumGothic.ttf')
    try:
        font = ImageFont.truetype(font_path, 66)
    except:
        font = ImageFont.load_default()
        st.error("❌ NanumGothic.ttf 폰트를 불러오지 못해 기본 폰트로 대체합니다.")

    # 이미지 불러오기 및 크기 통일
    images = [Image.open(file).convert("RGB") for file in file_names]
    width, height = images[0].size
    images = [img.resize((width, height)) for img in images]

    # 캡션 높이 추가
    caption_height = 90
    new_height = height + caption_height

    # 그리드 설정
    rows, cols = 2, 4
    grid_width = width * cols
    grid_height = new_height * rows

    # 새 이미지 생성
    combined_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined_image)

    # 이미지 및 캡션 붙이기
    for idx, (img, file_name) in enumerate(zip(images, file_names)):
        row = idx // cols
        col = idx % cols
        x = col * width
        y = row * new_height

        # 채널 이름 추출
        channel = file_name.split('_')[3]
        caption = channel_wavelength.get(channel, 'Unknown wavelength')

        # 캡션 텍스트 위치 계산
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (width - text_width) // 2
        text_y = y + 2.5  # 2.5px 아래 여백
        
        # 캡션 붙이기
        draw.text((text_x, text_y), caption, fill=(0, 0, 0), font=font)

        # 이미지 붙이기
        combined_image.paste(img, (x, y + caption_height))

    # 결합한 이미지 저장
    combined_image.save("combined_with_captions.jpg", quality=100)


def download_images(base_url, max_frames, save_dir):
    """
    지정된 URL 패턴과 프레임 수를 사용하여 이미지를 다운로드합니다.
    
    Args:
        base_url (str): 다운로드할 이미지의 URL 패턴.
        max_frames (int): 다운로드할 이미지의 총 프레임 수.
        save_dir (str): 이미지를 저장할 디렉토리 경로.
    
    Returns:
        tuple: 성공적으로 다운로드된 파일 수와 실패한 파일 수.
    """
    progress_placeholder = st.empty()
    progress_text = st.empty()
    progress_bar = progress_placeholder.progress(0)
    success_count = 0
    fail_count = 0

    progress_text.text("이미지 다운로드 중...")

    for i in range(max_frames):
        url = base_url.format(i)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = os.path.join(save_dir, f"ft{i:02d}.png")
                with open(filename, "wb") as f:
                    f.write(response.content)
                success_count += 1
            else:
                fail_count += 1
        except Exception:
            fail_count += 1

        # 프로그레스 바 업데이트
        progress_bar.progress((i + 1) / max_frames)

    progress_text.text("이미지 다운로드 완료!")

    # 다운로드 결과 반환
    return success_count, fail_count


# GIF 생성 함수
def create_gif(image_dir, gif_name):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    if image_files:
        gif_path = os.path.join(image_dir, gif_name)
        images = [Image.open(os.path.join(image_dir, f)) for f in image_files]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,  # 각 프레임 지속 시간 (밀리초)
            loop=0  # 무한 반복
        )
        return gif_path
    return None


# ===== 달 고도 계산 함수 =====
def calculate_moon_altitude(lat, lon):
    """달의 고도 계산 및 시각화"""
    # Skyfield 데이터 로드
    planets = load('de421.bsp')
    earth = planets['earth']
    
    # 위도/경도 값을 N/S, E/W로 처리
    lat_str = f"{abs(lat):.5f} {'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.5f} {'E' if lon >= 0 else 'W'}"

    observer = earth + Topos(lat_str, lon_str)

    # 시간 범위 설정
    ts = load.timescale()
    start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)

    # 10분 간격으로 시간 리스트 생성
    times = [ts.utc(start_time.year, start_time.month, start_time.day, h, m)
            for h in range(24) for m in range(0, 60, 10)]

    # 달 고도 계산
    altitudes = []
    for t in times:
        astrometric = observer.at(t).observe(planets['moon'])
        alt, az, d = astrometric.apparent().altaz()
        altitudes.append(alt.degrees)

    # 현재 시간의 달 고도 계산
    current_time = datetime.now(timezone.utc)
    current_t = ts.utc(current_time.year, current_time.month, current_time.day,
                    current_time.hour, current_time.minute, current_time.second)

    astrometric_now = observer.at(current_t).observe(planets['moon'])
    alt_now, az_now, d_now = astrometric_now.apparent().altaz()
    moon_altitude_now = alt_now.degrees

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # 달의 고도 변화 라인
    hours = np.linspace(-1, 1, len(altitudes))
    ax.plot(hours, altitudes, color="deepskyblue", linewidth=2, alpha=0.7)

    # 현재 고도 강조
    current_x = np.interp(current_time.hour + current_time.minute / 60, np.linspace(0, 24, len(altitudes)), hours)
    ax.scatter(current_x, moon_altitude_now, color="deepskyblue", s=200, edgecolor="white", lw=2, alpha=0.8)

    # 지평선 및 강조 요소
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.scatter(current_x, 0, color="deepskyblue", s=50, edgecolor="white", lw=0, alpha=1)
    ax.vlines(current_x, 0, moon_altitude_now, color="deepskyblue", linestyle="--", linewidth=2, alpha=0.8)

    # 한국 시간(KST) 변환 및 표기
    current_time_kst = current_time + timedelta(hours=9)
    current_time_str = current_time_kst.strftime("%H:%M")
    ax.text(
        current_x, -18, f"{current_time_str}\n{moon_altitude_now:.1f}°", ha="center", fontsize=12, color="white", fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3", alpha=0.7)
    )

    # 달이 지평선과 만나는 교차점 (출/몰 시간) 계산
    for i in range(1, len(altitudes)):
        if altitudes[i - 1] * altitudes[i] < 0:
            # 선형 보간으로 정확한 지점 추정
            x0, x1 = hours[i - 1], hours[i]
            y0, y1 = altitudes[i - 1], altitudes[i]
            crossing_x = x0 - y0 * (x1 - x0) / (y1 - y0)

            # 교차 시간 계산 (보간)
            total_minutes = (crossing_x + 1) * 720  # -1~1 => 0~1440분
            crossing_time = start_time + timedelta(minutes=total_minutes)
            crossing_time_kst = crossing_time + timedelta(hours=9)
            label = crossing_time_kst.strftime("%H:%M")

            # 출력 위치 조정
            label_y = 5
            label_text = "월출" if y1 > y0 else "월몰"
            ax.text(crossing_x, label_y, f"{label_text}\n{label}", ha="center", fontsize=12,
                    color="lightgreen" if y1 > y0 else "lightcoral",
                    bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3", alpha=0.7))

    # 축 제거
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return fig


def calculate_sun_altitude(lat, lon):
    """태양의 고도 계산 및 시각화"""
    # Skyfield 데이터 로드
    planets = load('de421.bsp')
    earth = planets['earth']
    
    # 위도/경도 값을 N/S, E/W로 처리
    lat_str = f"{abs(lat):.5f} {'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.5f} {'E' if lon >= 0 else 'W'}"

    observer = earth + Topos(lat_str, lon_str)

    # 시간 범위 설정
    ts = load.timescale()
    start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)

    # 10분 간격으로 시간 리스트 생성
    times = [ts.utc(start_time.year, start_time.month, start_time.day, h, m)
            for h in range(24) for m in range(0, 60, 10)]

    # 태양 고도 계산
    altitudes = []
    for t in times:
        astrometric = observer.at(t).observe(planets['sun'])
        alt, az, d = astrometric.apparent().altaz()
        altitudes.append(alt.degrees)

    # 현재 시간의 태양 고도 계산
    current_time = datetime.now(timezone.utc)
    current_t = ts.utc(current_time.year, current_time.month, current_time.day,
                    current_time.hour, current_time.minute, current_time.second)

    astrometric_now = observer.at(current_t).observe(planets['sun'])
    alt_now, az_now, d_now = astrometric_now.apparent().altaz()
    moon_altitude_now = alt_now.degrees

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # 탸양의 고도 변화 라인
    hours = np.linspace(-1, 1, len(altitudes))
    ax.plot(hours, altitudes, color="orange", linewidth=2, alpha=0.7)

    # 현재 고도 강조
    current_x = np.interp(current_time.hour + current_time.minute / 60, np.linspace(0, 24, len(altitudes)), hours)
    ax.scatter(current_x, moon_altitude_now, color="gold", s=200, edgecolor="white", lw=2, alpha=0.8)

    # 지평선 및 강조 요소
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.scatter(current_x, 0, color="orange", s=50, edgecolor="white", lw=0, alpha=1)
    ax.vlines(current_x, 0, moon_altitude_now, color="orange", linestyle="--", linewidth=2, alpha=0.8)

    # 한국 시간(KST) 변환 및 표기
    current_time_kst = current_time + timedelta(hours=9)
    current_time_str = current_time_kst.strftime("%H:%M")
    ax.text(
        current_x, -18, f"{current_time_str}\n{moon_altitude_now:.1f}°", ha="center", fontsize=12, color="white", fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3", alpha=0.7)
    )
    
    # 태양이 지평선과 만나는 교차점 (출/몰 시간) 계산
    for i in range(1, len(altitudes)):
        if altitudes[i - 1] * altitudes[i] < 0:
            # 선형 보간으로 정확한 지점 추정
            x0, x1 = hours[i - 1], hours[i]
            y0, y1 = altitudes[i - 1], altitudes[i]
            crossing_x = x0 - y0 * (x1 - x0) / (y1 - y0)

            # 교차 시간 계산 (보간)
            total_minutes = (crossing_x + 1) * 720  # -1~1 => 0~1440분
            crossing_time = start_time + timedelta(minutes=total_minutes)
            crossing_time_kst = crossing_time + timedelta(hours=9)
            label = crossing_time_kst.strftime("%H:%M")

            # 출력 위치 조정
            label_y = 5
            label_text = "일출" if y1 > y0 else "일몰"
            ax.text(crossing_x, label_y, f"{label_text}\n{label}", ha="center", fontsize=12,
                    color="lightgreen" if y1 > y0 else "lightcoral",
                    bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3", alpha=0.7))
            
    # 축 제거
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return fig


# ===== 기상 정보 파싱 함수 =====
def parse_weather_xml(xml_text):
    root = ET.fromstring(xml_text)
    items = root.findall(".//item")

    data = []
    for item in items:
        parsed = {
            "baseDate": item.findtext("baseDate"),
            "baseTime": item.findtext("baseTime"),
            "category": item.findtext("category"),
            "fcstDate": item.findtext("fcstDate"),
            "fcstTime": item.findtext("fcstTime"),
            "fcstValue": item.findtext("fcstValue"),
            "obsrValue": item.findtext("obsrValue")  # 초단기실황용
        }
        data.append(parsed)

    df = pd.DataFrame(data)
    return df


# ===== 천체 관측 가능지수 계산 함수 =====
def calculate_observation_quality(PTY, SQM, cloud_amount, humidity, moonphase, visibility):
    """
    관측 지수(COI) 및 가중치 항목들을 계산하는 함수.
    PTY가 0(강수 없음)이 아닐 경우 '관측불가' 반환.

    Returns:
        dict: {'관측불가'} 또는 {'COI': float, '가중치': {...}}
    """
    if PTY != 0:
        return {"결과": "관측불가"}

    # --- 가중치 계산 ---
    W_sqm = max(0, min((SQM - 18) / 4, 1))  # 18~22 기준으로 정규화
    W_cloud = (1 - cloud_amount / 100) ** 1.5
    W_humidity = 1 - 0.3 * (humidity / 100)
    W_moon = 1 - 0.7 * (moonphase / 100)
    W_visibility = min(visibility / 20000, 1.0)

    # --- COI 계산 ---
    W_total = W_sqm * W_cloud * W_humidity * W_moon * W_visibility
    COI = 1 + 8 * (1 - W_total)

    # --- 결과 반환 ---
    return {
        "COI": round(COI, 2),
        "가중치": {
            "W_광공해(Bortle)": round(W_sqm, 3),
            "W_구름량": round(W_cloud, 3),
            "W_습도": round(W_humidity, 3),
            "W_달위상": round(W_moon, 3),
            "W_대기시정": round(W_visibility, 3),
            "W_전체": round(W_total, 3)
        }
    }


def display_observation_quality(df_now, sqm, cloud_amount, moon_phase, visibility):
    """
    관측 지수를 계산하고 결과를 출력하는 함수.
    """
    # PTY와 REH 값 추출
    pty_val = int(df_now[df_now['category'] == 'PTY'].iloc[0]['obsrValue'])
    reh_val = int(df_now[df_now['category'] == 'REH'].iloc[0]['obsrValue'])

    # 관측 지수 계산
    result = calculate_observation_quality(
        PTY=pty_val,
        SQM=sqm,
        cloud_amount=cloud_amount,
        humidity=reh_val,
        moonphase=moon_phase,
        visibility=visibility
    )

    # 결과 출력
    if result.get("결과") == "관측불가":
        st.error("관측불가: 강수가 감지되었습니다.")
    else:
        coi = result["COI"]
        # 색상 매핑 (1~9)
        coi_colors = {
            1: "#4CAF50",  # 초록
            2: "#66BB6A",  # 연한 초록
            3: "#8BC34A",  # 라임
            4: "#CDDC39",  # 연한 라임
            5: "#FFEB3B",  # 노랑
            6: "#FFC107",  # 주황
            7: "#FF9800",  # 진한 주황
            8: "#F44336",  # 빨강
            9: "#D32F2F"   # 진한 빨강
        }
        coi_color = coi_colors.get(int(coi), "#FFFFFF")  # 기본값 흰색

        # COI 결과 출력
        st.markdown(
            f"""
            <div style="
                background-color: {coi_color};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                font-size: 24px;
                font-weight: bold;
            ">
                천체관측 가능 지수 (COI): {coi}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # 가중치 정보를 Columns로 나누어 표시
        weights = result["가중치"]
        weight_keys = list(weights.keys())
        cols = st.columns(len(weight_keys))

        for i, key in enumerate(weight_keys):
            cols[i].metric(label=key, value=weights[key])


# ===== 메인 애플리케이션 =====
def main():
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="천체 관측 가능지수 측량기",
        layout="wide"
    )

    st.title("🌐 천체 관측 가능지수 측량기")
    st.subheader("이 웹 애플리케이션은 천체 관측을 위한 다양한 기능을 제공합니다.")

    # 한글 폰트 설정 적용
    setup_korean_font()

    # UI 구성: 탭 인터페이스 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📍 사용자 위치 지정", 
        "🛰️ 위성 이미지", 
        "🌙 태양/달 고도 그래프",
        "🌦️ 각종 기상 정보", 
        "🔭 천체 관측 가능지수(COI)"
    ])

    # 탭 1: 사용자 위치 지정
    with tab1:
        st.markdown("""            
        지도에서 원하는 위치를 클릭하여 위도와 경도, SQM값을 확인하세요. 클릭한 위치의 위도와 경도, SQM값을 아래 입력란에 입력하거나, 기본값(전북과학고등학교 천문대)을 사용할 수 있습니다.
        """)

        col1, col2 = st.columns([1,3])

        with col1:
            # 위치 선택 방식
            location_method = st.radio("위치 선택 방법", ["고정된 지점 선택", "직접 입력"], index=0)
        
        predefined_locations = {
                    "전북과학고등학교 천문대": ("36.01406", "127.03570", "20.46"),
                    # 향후 여기에 더 추가 가능
                }
        
        if location_method == "고정된 지점 선택":
            with col2:
                place_name = st.selectbox("지점을 선택하세요", list(predefined_locations.keys()))
                user_lat, user_lon, sqm = map(float, predefined_locations[place_name])

        else:
            
            with col2:
                col1, col2, col3 = st.columns(3)

                with col1:
                    user_lat = st.number_input("위도 (Latitude)", value=36.50000, format="%.5f", key="input_lat")

                with col2:
                    user_lon = st.number_input("경도 (Longitude)", value=127.50000, format="%.5f", key="input_lon")

                with col3:
                    sqm = st.number_input("SQM", value=20.50, format="%.2f", key="input_sqm")

            # 입력값을 세션에 저장
            st.session_state.user_lat = user_lat
            st.session_state.user_lon = user_lon
            st.session_state.sqm = sqm

        st.info(f"사용 중인 위치의 값: 위도 {user_lat}, 경도 {user_lon}, SQM {sqm}")

        # iframe 삽입
        iframe_code = """
        <iframe src="https://www.lightpollutionmap.info/#zoom=5&lat=36.5&lon=127.5&layers=B0FFFFFFFT"
        width="100%" height="550" style="border:none;"></iframe>
        """
        st.components.v1.html(iframe_code, height=600, scrolling=True)

        # 값 유지 위해 세션 상태 저장
        st.session_state.user_lat = user_lat
        st.session_state.user_lon = user_lon
        st.session_state.sqm = sqm

    # 탭 2: 위성 이미지
    with tab2:
        file_names, channel_wavelength = download_satellite_images()
        combine_satellite_images(file_names, channel_wavelength)
        
        # 이미지 렌더링
        image_path = "combined_with_captions.jpg"
        render_img_html(image_to_base64(image_path))

    # 탭 3: 태양/달 고도 그래프
    with tab3:
        st.markdown("입력한 위치에서의 달과 태양의 고도 변화를 확인할 수 있습니다.")
        
        moon_altitude_fig = calculate_moon_altitude(user_lat, user_lon)
        sun_altitude_fig = calculate_sun_altitude(user_lat, user_lon)

        col1_1, col2_2 = st.columns(2)
       
        with col1_1:
            st.subheader("🌙 달 고도 변화")
            st.pyplot(moon_altitude_fig)

            st.subheader("☀️ 태양 고도 변화")
            st.pyplot(sun_altitude_fig)

        with col2_2:
            st.subheader("오늘의 달 정보")

            # 현재 날짜 가져오기
            today = datetime.today()
            year = today.strftime("%Y")
            month = today.strftime("%m")
            day = today.strftime("%d")

            # API 요청 URL 및 파라미터 설정
            url = 'http://apis.data.go.kr/B090041/openapi/service/LunPhInfoService/getLunPhInfo'
            params = {
                'serviceKey': 'CIixk5nqh86hsMOFj1C3YPp4LxndXeRB848pZgiKuSmSbbLRBgfZqAReCjaDxDdfi2q8GW5N1Z0+ilyWpN4Epg==',
                'solYear': year,
                'solMonth': month,
                'solDay': day
            }

            # API 요청
            response = requests.get(url, params=params)

            # XML 파싱
            root = ET.fromstring(response.content)

            # 월령(lunAge) 값 추출
            lunAge = float(root.find(".//lunAge").text)  # XPath를 이용한 접근

            def moon_phase_percentage(moon_age):
                # 월령을 기준으로 위상(%) 계산
                if moon_age < 7.4:
                    # 초생달에서 상현달로 가는 구간 (0% ~ 50%)
                    phase = (moon_age / 7.4) * 50
                elif moon_age < 14.8:
                    # 상현달에서 보름달로 가는 구간 (50% ~ 100%)
                    phase = 50 + ((moon_age - 7.4) / 7.4) * 50
                elif moon_age < 22.1:
                    # 보름달에서 하현달로 가는 구간 (100% ~ 50%)
                    phase = 100 - ((moon_age - 14.8) / 7.4) * 50
                else:
                    # 하현달에서 초생달로 가는 구간 (50% ~ 0%)
                    phase = 50 - ((moon_age - 22.1) / 7.4) * 50

                return phase

            # 예시: 월령을 사용하여 달의 위상(%) 계산
            moon_age = lunAge  # 월령 (예: 10일)
            moon_phase = moon_phase_percentage(moon_age)

            # 월령 기반 이미지 경로 계산
            moon_day = int(round(moon_age))  # 가장 가까운 정수로 반올림
            moon_image_path = os.path.join(os.path.dirname(__file__), 'moon', f'Day {moon_day}.jpg')
                
            # 이미지 로드 및 출력
            if os.path.exists(moon_image_path):
                image = Image.open(moon_image_path)
                st.image(image, use_container_width=True)
            else:
                st.warning(f"이미지 {moon_image_path} 를 찾을 수 없습니다.")

            st.markdown(f"""
                <div style="
                    background-color: #e6f0fa;
                    padding: 15px;
                    border-left: 5px solid #91c6f8;
                    border-radius: 5px;
                    text-align: center;
                    font-size: 18px;
                    color: #003366;
                ">
                    월령 <b>{moon_age:.1f}일</b> / 달의 위상 <b>{moon_phase:.1f}%</b>
                </div>
            """, unsafe_allow_html=True)

    # 탭 4: 각종 기상 정보
    with tab4:
        # 등급과 색상 구하기
        def get_air_quality_level(value, pollutant):
            if pollutant == "PM10":
                if value <= 30:
                    return "좋음", "#4FC3F7"
                elif value <= 80:
                    return "보통", "#81C784"
                elif value <= 150:
                    return "나쁨", "#FFF176"
                else:
                    return "매우나쁨", "#E57373"

            elif pollutant == "PM2.5":
                if value <= 15:
                    return "좋음", "#4FC3F7"
                elif value <= 35:
                    return "보통", "#81C784"
                elif value <= 75:
                    return "나쁨", "#FFF176"
                else:
                    return "매우나쁨", "#E57373"

            elif pollutant == "O3":
                if value <= 0.03:
                    return "좋음", "#4FC3F7"
                elif value <= 0.09:
                    return "보통", "#81C784"
                elif value <= 0.15:
                    return "나쁨", "#FFF176"
                else:
                    return "매우나쁨", "#E57373"

            return "정보 없음", "#CCCCCC"

        # TM 좌표 -> 측정소 -> 데이터 표시
        tmX, tmY = convert_to_tm(user_lat, user_lon)
        station_name = get_nearest_station(tmX, tmY)

        if station_name:
            st.success(f"가장 가까운 미세먼지 측정소: **{station_name}**")

            pm10, pm25, o3 = get_air_quality(station_name)

            if pm10 and pm25 and o3:
                # 문자열 -> 숫자 변환
                pm10_val = float(pm10)
                pm25_val = float(pm25)
                o3_val = float(o3)

                # 등급과 색상 가져오기
                pm10_level, pm10_color = get_air_quality_level(pm10_val, "PM10")
                pm25_level, pm25_color = get_air_quality_level(pm25_val, "PM2.5")
                o3_level, o3_color = get_air_quality_level(o3_val, "O3")

                # 컬럼별 표시
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f'<div style="background-color:{pm10_color}; padding: 10px; border-radius: 10px; text-align:center;">'
                        f'<b>PM10 (미세먼지)</b><br>{pm10_val} µg/m³<br><b>{pm10_level}</b>'
                        '</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown(
                        f'<div style="background-color:{pm25_color}; padding: 10px; border-radius: 10px; text-align:center;">'
                        f'<b>PM2.5 (초미세먼지)</b><br>{pm25_val} µg/m³<br><b>{pm25_level}</b>'
                        '</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown(
                        f'<div style="background-color:{o3_color}; padding: 10px; border-radius: 10px; text-align:center;">'
                        f'<b>오존 (O<sub>3</sub>)</b><br>{o3_val:.3f} ppm<br><b>{o3_level}</b>'
                        '</div>', unsafe_allow_html=True)
            else:
                st.error("미세먼지 정보를 불러오지 못했습니다.")
        else:
            st.error("근처 측정소를 찾을 수 없습니다.")

        st.divider()

        st.subheader("🌤️ 현재 기상 정보")
        
        api_url = 'https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_xy_lonlat'
        api_parameters = {
            'lon': user_lon,
            'lat': user_lat,
            'help': 1,
            'authKey': API_KEY
            }

        response = requests.get(api_url, params=api_parameters)

        if response.status_code != 200 or len(response.content) < 100:
            st.error("❌ 격자 좌표 요청 실패")
        else:
                # content 디코딩 후 StringIO로 메모리 처리
            decoded = response.content.decode('euc-kr')
            text_stream = io.StringIO(decoded)

            # 줄별로 읽기
            lines = text_stream.readlines()
            data_line = lines[-1].strip()
            lon_val, lat_val, nx, ny = map(str.strip, data_line.split(","))

        # 초단기실황
        base_date, base_time, res = call_weather_api(
            '초단기실황', 
            'https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtNcst',
            nx, ny
        )
        df_now = parse_weather_xml(res.text)

        # 초단기예보
        base_date, base_time, res = call_weather_api(
            '초단기예보', 
            'https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtFcst',
            nx, ny
        )
        df_fcst = parse_weather_xml(res.text)

        # SKY 값 추출 (초단기예보에서)
        sky_row = df_fcst[df_fcst['category'] == 'SKY'].iloc[0]
        sky_val = sky_row['fcstValue']

        # SKY 코드 -> 설명
        sky_mapping = {
            '1': '맑음', '3': '구름많음', '4': '흐림'
        }
        sky_desc = sky_mapping.get(str(int(float(sky_val))), '정보 없음')

        # PTY 코드 -> 설명
        pty_mapping = {
            '0': '없음', '1': '비', '2': '비/눈', '3': '눈',
            '4': '소나기', '5': '빗방울', '6': '빗방울눈날림', '7': '눈날림'
        }

        # 시각화할 항목 선택 (표시 순서)
        visual_keys = ['T1H', 'REH', 'WSD', 'SKY', 'PTY', 'RN1', 'SEE', 'CLD']

        # 라벨 (이모지 포함)
        labels = {
            'T1H': '🌡️ 기온 (℃)',
            'RN1': '🌧️ 1시간 강수량 (mm)',
            'REH': '💧 습도 (%)',
            'PTY': '🌂 강수형태',
            'SKY': '🌤️ 하늘 상태',
            'WSD': '🌬️ 풍속 (m/s)',
            'SEE': '🌫️ 가시거리 (m)',
            'CLD': '☁️ 구름량 (%)'
        }

        # 실황 데이터 가져오기
        data = {row['category']: row['obsrValue'] for _, row in df_now.iterrows()}

        # SKY 추가 (초단기예보)
        data['SKY'] = sky_desc

        # 현재 한국 시각 구하기
        kst = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst)

        # 현재 시간을 15분 단위로 내림 처리
        rounded_time = now_kst.replace(
            minute=(now_kst.minute // 15) * 15, second=0, microsecond=0
        )
        now_str = rounded_time.strftime("%Y-%m-%d %H:%M")

        # API 호출
        url = "https://my.meteoblue.com/packages/clouds-15min"
        
        api_parameters = {
            'apikey': 'wOV0tPijaQGbhC51',
            'lat': user_lat,
            'lon': user_lon,
            'asl': 34,
            'format': 'json'
            }

        response = requests.get(url, params=api_parameters)
        data = response.json()

        # 데이터 확인 및 값 추출
        time_list = data["data_xmin"]["time"]
        idx = time_list.index(now_str) if now_str in time_list else -1
        cloud_amount = int(data["data_xmin"]["totalcloudcover"][idx])
        visibility = int(data["data_xmin"]["visibility"][idx])

        data['SEE'] = visibility
        data['CLD'] = cloud_amount

        # PTY 설명 치환
        if 'PTY' in data:
            data['PTY'] = pty_mapping.get(str(int(float(data['PTY']))), '정보 없음')

        # 두 줄로 나눠서 시각화
        cols1 = st.columns(4)
        cols2 = st.columns(4)

        for idx, key in enumerate(visual_keys):
            value = data.get(key, "N/A")
            label = labels.get(key, key)
            col = cols1[idx] if idx < 4 else cols2[idx - 4]
            col.metric(label=label, value=value)

        st.divider()

        ch_save_dir = "ch_weather_images"
        cml_save_dir = "cml_weather_images"
        os.makedirs(ch_save_dir, exist_ok=True)
        os.makedirs(cml_save_dir, exist_ok=True)

        # 기본 URL과 프레임 수를 변수로 저장
        ch_url = "https://tingala.net/gpv-map/map/msm/ch/ft{:02d}.png"
        cml_url = "https://tingala.net/gpv-map/map/msm/cml/ft{:02d}.png"
        max_frames = 51  # 다운로드할 프레임 수
        
        # 이미지 다운로드
        download_images(ch_url, max_frames, ch_save_dir)
        download_images(cml_url, max_frames, cml_save_dir)

        # GIF 생성
        ch_gif_path = create_gif(ch_save_dir, "ch_weather_animation.gif")
        cml_gif_path = create_gif(cml_save_dir, "cml_weather_animation.gif")

        # GIF 표시
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("상층 구름 예보(일/시간)")
            if ch_gif_path:
                st.image(ch_gif_path, caption="상층 구름 예보 애니메이션", use_container_width=True)
            else:
                st.warning(f"{ch_save_dir} 폴더에 PNG 이미지가 없습니다. '이미지 다운로드' 탭에서 먼저 이미지를 다운로드하세요.")

        with col2:
            st.subheader("중상층 구름 예보(일/시간)")
            if cml_gif_path:
                st.image(cml_gif_path, caption="중하층 구름 예보 애니메이션", use_container_width=True)
            else:
                st.warning(f"{cml_save_dir} 폴더에 PNG 이미지가 없습니다. '이미지 다운로드' 탭에서 먼저 이미지를 다운로드하세요.")

    # 탭 5: 천체 관측 가능지수
    with tab5:
        # 관측 지수 계산 및 출력
        display_observation_quality(df_now, sqm, cloud_amount, moon_phase, visibility)

        st.divider()

        # 설명 텍스트
        explanation = (
            "다섯 번째 탭에서는 천체 관측 가능지수(COI)를 계산합니다. "
            "COI는 광공해, 구름량, 습도, 달 위상, 대기 시정과 같은 요소를 기반으로 계산되며, "
            "각 요소는 가중치로 변환됩니다. 가중치는 정규화된 수식으로 계산되며, "
            "최종적으로 모든 가중치를 곱하여 전체 가중치(W_total)를 구합니다. "
            "COI는 이 값을 기반으로 1에서 9 사이의 값으로 계산되며, 값이 낮을수록 관측 가능성이 높음을 의미합니다. "
            "강수가 감지되면 관측 불가로 표시됩니다."
        )

        # 텍스트 출력
        st.text(explanation)

        # COI 설명 표
        coi_table = """
        <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 14px;">
            <thead>
            <tr style="background-color: #007acc; color: white;">
            <th style="padding: 10px; border: 1px solid #ddd;">COI 값</th>
            <th style="padding: 10px; border: 1px solid #ddd;">설명</th>
            </tr>
            </thead>
            <tbody>
            <tr style="background-color: #4CAF50; color: white;">
            <td style="padding: 10px; border: 1px solid #ddd;">1</td>
            <td style="padding: 10px; border: 1px solid #ddd;">최적의 관측 조건</td>
            </tr>
            <tr style="background-color: #66BB6A; color: white;">
            <td style="padding: 10px; border: 1px solid #ddd;">2</td>
            <td style="padding: 10px; border: 1px solid #ddd;">매우 좋은 관측 조건</td>
            </tr>
            <tr style="background-color: #8BC34A; color: white;">
            <td style="padding: 10px; border: 1px solid #ddd;">3</td>
            <td style="padding: 10px; border: 1px solid #ddd;">좋은 관측 조건</td>
            </tr>
            <tr style="background-color: #CDDC39; color: black;">
            <td style="padding: 10px; border: 1px solid #ddd;">4</td>
            <td style="padding: 10px; border: 1px solid #ddd;">관측 가능</td>
            </tr>
            <tr style="background-color: #FFEB3B; color: black;">
            <td style="padding: 10px; border: 1px solid #ddd;">5</td>
            <td style="padding: 10px; border: 1px solid #ddd;">보통</td>
            </tr>
            <tr style="background-color: #FFC107; color: black;">
            <td style="padding: 10px; border: 1px solid #ddd;">6</td>
            <td style="padding: 10px; border: 1px solid #ddd;">관측 어려움</td>
            </tr>
            <tr style="background-color: #FF9800; color: white;">
            <td style="padding: 10px; border: 1px solid #ddd;">7</td>
            <td style="padding: 10px; border: 1px solid #ddd;">관측 매우 어려움</td>
            </tr>
            <tr style="background-color: #F44336; color: white;">
            <td style="padding: 10px; border: 1px solid #ddd;">8</td>
            <td style="padding: 10px; border: 1px solid #ddd;">관측 불가능에 가까움</td>
            </tr>
            <tr style="background-color: #D32F2F; color: white;">
            <td style="padding: 10px; border: 1px solid #ddd;">9</td>
            <td style="padding: 10px; border: 1px solid #ddd;">관측 불가능</td>
            </tr>
            </tbody>
        </table>
        """

        st.markdown(coi_table, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
