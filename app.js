import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/+esm";

// ============================================================
// 0. 설정
// ============================================================

const DEFAULT_QUALITIES = [
  "maj", "m", "7", "maj7", "m7", "6", "m6", "9",
  "maj9", "m9", "add9", "7sus4", "dim7", "m7b5",
  "7b9", "maj7#11"
];

const DEFAULT_SETTINGS = {
  mode: "basic",
  slotCount: 8,
  qualityCount: 16,
  qualities: [...DEFAULT_QUALITIES],
  manualChords: [
    "Cmaj7", "Dm7", "Em7", "Fmaj7",
    "G7", "Am7", "Bdim", "Cadd9"
  ],
  tone: 25,
  cameraFacing: "user",
  calibration: {
    left:  { closed: 0.28, open: 0.86 },
    right: { closed: 0.28, open: 0.86 }
  }
};

const STORAGE_KEY = "handChordV5Settings";
const ROOTS = ["C", "D", "E", "F", "G", "A", "B"];

let settings = loadSettings();

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && typeof saved === "object") {
      return {
        ...DEFAULT_SETTINGS,
        ...saved,
        qualities: Array.isArray(saved.qualities)
          ? saved.qualities : [...DEFAULT_SETTINGS.qualities],
        manualChords: Array.isArray(saved.manualChords)
          ? saved.manualChords : [...DEFAULT_SETTINGS.manualChords],
        calibration: {
          left: {
            ...DEFAULT_SETTINGS.calibration.left,
            ...(saved.calibration?.left || {})
          },
          right: {
            ...DEFAULT_SETTINGS.calibration.right,
            ...(saved.calibration?.right || {})
          }
        }
      };
    }
  } catch (_) {}
  return JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
}

function persistSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

// ============================================================
// 1. DOM
// ============================================================

const video = document.querySelector("#video");
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");

const currentChordEl = document.querySelector("#currentChord");
const statusEl = document.querySelector("#status");
const engineStatusEl = document.querySelector("#engineStatus");
const startBtn = document.querySelector("#startBtn");
const settingsBtn = document.querySelector("#settingsBtn");

const leftOpenFill = document.querySelector("#leftOpenFill");
const leftOpenValue = document.querySelector("#leftOpenValue");
const rightOpenFill = document.querySelector("#rightOpenFill");
const rightOpenValue = document.querySelector("#rightOpenValue");

const settingsPanel = document.querySelector("#settingsPanel");
const closeSettingsBtn = document.querySelector("#closeSettingsBtn");
const saveSettingsBtn = document.querySelector("#saveSettingsBtn");
const resetSettingsBtn = document.querySelector("#resetSettingsBtn");

const modeSelect = document.querySelector("#modeSelect");
const basicSettings = document.querySelector("#basicSettings");
const manualSettings = document.querySelector("#manualSettings");

const qualityCountSelect = document.querySelector("#qualityCount");
const qualityInputs = document.querySelector("#qualityInputs");

const slotCountSelect = document.querySelector("#slotCount");
const manualInputs = document.querySelector("#manualInputs");

const toneSlider = document.querySelector("#toneSlider");
const cameraFacingSelect = document.querySelector("#cameraFacingSelect");

const leftRawValue = document.querySelector("#leftRawValue");
const rightRawValue = document.querySelector("#rightRawValue");
const leftClosedSaved = document.querySelector("#leftClosedSaved");
const leftOpenSaved = document.querySelector("#leftOpenSaved");
const rightClosedSaved = document.querySelector("#rightClosedSaved");
const rightOpenSaved = document.querySelector("#rightOpenSaved");
const leftClosedCalBtn = document.querySelector("#leftClosedCalBtn");
const leftOpenCalBtn = document.querySelector("#leftOpenCalBtn");
const rightClosedCalBtn = document.querySelector("#rightClosedCalBtn");
const rightOpenCalBtn = document.querySelector("#rightOpenCalBtn");
const calibrationStatus = document.querySelector("#calibrationStatus");


// ============================================================
// 2. MediaPipe
// ============================================================

let handLandmarker = null;
let activeStream = null;
let cameraStarted = false;

let handEngineName = "";
let lastVideoTime = -1;
let lastDetectAt = 0;
let cachedHands = [];
let detectErrorCount = 0;

function isMobileDevice() {
  return (
    /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
    (navigator.maxTouchPoints || 0) > 1
  );
}

async function makeHandLandmarker(vision, useGpu) {
  const baseOptions = {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  };

  // 모바일은 GPU delegate가 기기/브라우저에 따라 아예 손 검출이 안 되는 경우가 있어
  // CPU(WASM)를 기본으로 사용한다.
  if (useGpu) {
    baseOptions.delegate = "GPU";
  }

  return await HandLandmarker.createFromOptions(vision, {
    baseOptions,
    runningMode: "VIDEO",
    numHands: 2,

    // 모바일 카메라는 흔들림/노이즈가 커서 조금 완화
    minHandDetectionConfidence: 0.30,
    minHandPresenceConfidence: 0.30,
    minTrackingConfidence: 0.30
  });
}

async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
  );

  const mobile = isMobileDevice();

  if (mobile) {
    // 모바일: CPU 우선
    engineStatusEl.textContent = "손 인식 엔진 불러오는 중…";
    try {
      handLandmarker = await makeHandLandmarker(vision, false);
      handEngineName = "모바일 CPU";
    } catch (cpuError) {
      console.warn("CPU hand landmarker failed, trying GPU:", cpuError);
      handLandmarker = await makeHandLandmarker(vision, true);
      handEngineName = "모바일 GPU";
    }
  } else {
    // PC: GPU 우선, 실패 시 CPU
    try {
      handLandmarker = await makeHandLandmarker(vision, true);
      handEngineName = "GPU";
    } catch (gpuError) {
      console.warn("GPU hand landmarker failed, falling back to CPU:", gpuError);
      handLandmarker = await makeHandLandmarker(vision, false);
      handEngineName = "CPU";
    }
  }

  engineStatusEl.textContent = `손 인식: ${handEngineName}`;
}

// ============================================================
// 3. 범용 코드 파서
//
// 사용자 코드 모드에서 일반적인 코드 기호를 조합해서 쓸 수 있다.
//
// 예:
// C, Cm, C5, C6, Cm6, C7, Cmaj7, Cm7
// C9, Cmaj9, Cm9, C11, C13
// Cadd9, Cadd11, Cadd13
// Csus2, Csus4, C7sus4
// Cdim, Cdim7, Caug, Cm7b5
// C7b9, C7#9, C7b5, C7#5
// Cmaj7#11, C13b9, C7b9#11
// CmMaj7, C6/9, C/E, C7(b9,#11)
// ============================================================

const ROOT_SEMITONES = {
  C:0, "C#":1, Db:1, D:2, "D#":3, Eb:3, E:4,
  F:5, "F#":6, Gb:6, G:7, "G#":8, Ab:8,
  A:9, "A#":10, Bb:10, B:11
};

function normalizeChordText(text) {
  return String(text)
    .trim()
    .replaceAll("♭", "b")
    .replaceAll("♯", "#")
    .replaceAll("Δ", "maj")
    .replaceAll("−", "m")
    .replace(/\s+/g, "")
    .replace(/[(),]/g, "");
}

function uniqueSorted(arr) {
  return [...new Set(arr)].sort((a,b) => a-b);
}

function pc(n) {
  return ((n % 12) + 12) % 12;
}

function removePitchClasses(intervals, pcs) {
  return intervals.filter(x => !pcs.includes(pc(x)));
}

function replaceNaturalDegree(intervals, natural, altered) {
  intervals = intervals.filter(x => pc(x) !== pc(natural));
  intervals.push(altered);
  return intervals;
}

function parseChord(chordName) {
  let raw = normalizeChordText(chordName);
  if (!raw) return null;

  // 6/9는 slash bass로 오인되지 않도록 임시 표기
  raw = raw.replace(/6\/9/g, "69");

  // slash bass: C/E, Am/C, C69/E
  let slashBass = null;
  const slash = raw.match(/\/([A-Ga-g][#b]?)$/);
  if (slash) {
    slashBass = slash[1][0].toUpperCase() + slash[1].slice(1);
    if (!(slashBass in ROOT_SEMITONES)) return null;
    raw = raw.slice(0, -slash[0].length);
  }

  const rootMatch = raw.match(/^([A-Ga-g])([#b]?)(.*)$/);
  if (!rootMatch) return null;

  const root = rootMatch[1].toUpperCase() + (rootMatch[2] || "");
  if (!(root in ROOT_SEMITONES)) return null;

  let s = rootMatch[3] || "";

  // 흔한 별칭
  s = s
    .replace(/^minor/i, "m")
    .replace(/^min/i, "m")
    .replace(/^major/i, "maj")
    .replace(/^mM(?=\d)/, "mMaj")
    .replace(/^mmaj/i, "mMaj")
    .replace(/^ø7?/, "m7b5")
    .replace(/^°7/, "dim7")
    .replace(/^°/, "dim")
    .replace(/^\+/, "aug");

  let intervals = [0,4,7];
  let cursor = s;

  function consume(token) {
    if (cursor.startsWith(token)) {
      cursor = cursor.slice(token.length);
      return true;
    }
    return false;
  }

  // 기본 코드 성격 + 기본 확장
  if (consume("mMaj13"))      intervals = [0,3,7,11,14,17,21];
  else if (consume("mMaj11")) intervals = [0,3,7,11,14,17];
  else if (consume("mMaj9"))  intervals = [0,3,7,11,14];
  else if (consume("mMaj7"))  intervals = [0,3,7,11];
  else if (consume("m7b5"))   intervals = [0,3,6,10];
  else if (consume("dim7"))   intervals = [0,3,6,9];
  else if (consume("dim"))    intervals = [0,3,6];
  else if (consume("aug13"))  intervals = [0,4,8,10,14,17,21];
  else if (consume("aug11"))  intervals = [0,4,8,10,14,17];
  else if (consume("aug9"))   intervals = [0,4,8,10,14];
  else if (consume("aug7"))   intervals = [0,4,8,10];
  else if (consume("aug"))    intervals = [0,4,8];
  else if (consume("maj13"))  intervals = [0,4,7,11,14,17,21];
  else if (consume("maj11"))  intervals = [0,4,7,11,14,17];
  else if (consume("maj9"))   intervals = [0,4,7,11,14];
  else if (consume("maj7"))   intervals = [0,4,7,11];
  else if (consume("m13"))    intervals = [0,3,7,10,14,17,21];
  else if (consume("m11"))    intervals = [0,3,7,10,14,17];
  else if (consume("m9"))     intervals = [0,3,7,10,14];
  else if (consume("m7"))     intervals = [0,3,7,10];
  else if (consume("m69"))    intervals = [0,3,7,9,14];
  else if (consume("m6"))     intervals = [0,3,7,9];
  else if (consume("m"))      intervals = [0,3,7];
  else if (consume("69"))     intervals = [0,4,7,9,14];
  else if (consume("13"))     intervals = [0,4,7,10,14,17,21];
  else if (consume("11"))     intervals = [0,4,7,10,14,17];
  else if (consume("9"))      intervals = [0,4,7,10,14];
  else if (consume("7"))      intervals = [0,4,7,10];
  else if (consume("6"))      intervals = [0,4,7,9];
  else if (consume("5"))      intervals = [0,7];
  else if (consume("maj"))    intervals = [0,4,7];

  // 뒤에 붙는 수식어를 반복해서 처리
  let guard = 0;
  while (cursor.length && guard++ < 40) {
    // sus
    if (cursor.startsWith("sus2")) {
      intervals = removePitchClasses(intervals, [3,4]);
      intervals.push(2);
      cursor = cursor.slice(4);
      continue;
    }
    if (cursor.startsWith("sus4")) {
      intervals = removePitchClasses(intervals, [3,4]);
      intervals.push(5);
      cursor = cursor.slice(4);
      continue;
    }

    // add9 / add#11 / addb9 등
    const addMatch = cursor.match(/^add([b#]?)(2|4|6|9|11|13)/);
    if (addMatch) {
      const accidental = addMatch[1];
      const degree = addMatch[2];
      const natural = {2:2,4:5,6:9,9:14,11:17,13:21}[degree];
      const delta = accidental === "b" ? -1 : accidental === "#" ? 1 : 0;
      intervals.push(natural + delta);
      cursor = cursor.slice(addMatch[0].length);
      continue;
    }

    // no3 / omit5 등
    const omitMatch = cursor.match(/^(?:no|omit)(3|5|7|9|11|13)/);
    if (omitMatch) {
      const degree = omitMatch[1];
      const pcs = {
        3:[3,4],
        5:[6,7,8],
        7:[9,10,11],
        9:[1,2,3],
        11:[4,5,6],
        13:[8,9,10]
      }[degree];
      intervals = removePitchClasses(intervals, pcs);
      cursor = cursor.slice(omitMatch[0].length);
      continue;
    }

    // b5 / #5 / b9 / #9 / #11 / b13 등
    const altMatch = cursor.match(/^([b#])(5|9|11|13)/);
    if (altMatch) {
      const sign = altMatch[1];
      const degree = altMatch[2];
      const natural = {5:7,9:14,11:17,13:21}[degree];
      const altered = natural + (sign === "b" ? -1 : 1);
      intervals = replaceNaturalDegree(intervals, natural, altered);
      cursor = cursor.slice(altMatch[0].length);
      continue;
    }

    // alt: 실용적인 dominant altered 묶음
    if (cursor.startsWith("alt")) {
      intervals = [0,4,6,8,10,13,15];
      cursor = cursor.slice(3);
      continue;
    }

    // 여기까지 안 맞으면 알 수 없는 표기
    return null;
  }

  intervals = uniqueSorted(intervals);

  return { root, intervals, slashBass };
}

function chordToMidi(chordName) {
  const parsed = parseChord(chordName);
  if (!parsed) return null;

  const rootMidi = 48 + ROOT_SEMITONES[parsed.root]; // C3 기준
  let notes = parsed.intervals.map(i => rootMidi + i);

  if (parsed.slashBass) {
    let bass = 36 + ROOT_SEMITONES[parsed.slashBass];
    while (bass >= Math.min(...notes)) bass -= 12;
    notes.unshift(bass);
  }

  return [...new Set(notes)].sort((a,b) => a-b);
}

function midiToFreq(note) {
  return 440 * Math.pow(2, (note - 69) / 12);
}

function buildChordName(root, quality) {
  return quality === "maj" ? root : root + quality;
}

// ============================================================
// 4. 따뜻한 지속형 신디
// ============================================================

class WarmSynth {
  constructor() {
    this.audioCtx = null;
    this.master = null;
    this.compressor = null;

    // 채널별로 독립적인 코드를 유지한다.
    // basic  = 기본 양손 모드에서 만들어지는 하나의 코드
    // left   = 사용자 모드 왼손 코드
    // right  = 사용자 모드 오른손 코드
    this.channels = new Map();

    this.tone = settings.tone;
  }

  async init() {
    if (!this.audioCtx) {
      this.audioCtx = new AudioContext();

      this.compressor = this.audioCtx.createDynamicsCompressor();
      this.compressor.threshold.value = -22;
      this.compressor.knee.value = 18;
      this.compressor.ratio.value = 3;
      this.compressor.attack.value = 0.012;
      this.compressor.release.value = 0.30;

      this.master = this.audioCtx.createGain();
      this.master.gain.value = 0.82;

      this.master.connect(this.compressor);
      this.compressor.connect(this.audioCtx.destination);
    }

    if (this.audioCtx.state === "suspended") {
      await this.audioCtx.resume();
    }
  }

  setTone(value) {
    this.tone = Number(value);
  }

  playChord(chordName, channel = "basic") {
    if (!this.audioCtx) return;

    const notes = chordToMidi(chordName);
    if (!notes) return;

    const old = this.channels.get(channel);

    // 같은 손/채널에서 같은 코드면 새로 만들지 않는다.
    if (old && old.chordName === chordName) return;

    // 이 손이 이전에 잡고 있던 코드만 release.
    this.stopChannel(channel, 0.24);

    const now = this.audioCtx.currentTime;
    const voices = [];
    const cutoff = 760 + (this.tone / 100) * 2100;

    // 두 코드를 동시에 울릴 때 너무 커지지 않도록 음 수에 따라 자동 보정.
    const targetGain = Math.max(
      0.018,
      0.042 - Math.max(0, notes.length - 3) * 0.0032
    );

    for (const midi of notes) {
      const filter = this.audioCtx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(cutoff, now);
      filter.Q.value = 0.35;

      const noteGain = this.audioCtx.createGain();
      noteGain.gain.setValueAtTime(0.0001, now);
      noteGain.gain.exponentialRampToValueAtTime(targetGain, now + 0.11);

      filter.connect(noteGain);
      noteGain.connect(this.master);

      const osc1 = this.audioCtx.createOscillator();
      osc1.type = "sine";
      osc1.frequency.value = midiToFreq(midi);
      osc1.detune.value = -3;

      const osc2 = this.audioCtx.createOscillator();
      osc2.type = "triangle";
      osc2.frequency.value = midiToFreq(midi);
      osc2.detune.value = 3;

      const g1 = this.audioCtx.createGain();
      const g2 = this.audioCtx.createGain();
      g1.gain.value = 0.80;
      g2.gain.value = 0.20;

      osc1.connect(g1);
      osc2.connect(g2);
      g1.connect(filter);
      g2.connect(filter);

      osc1.start(now);
      osc2.start(now);

      voices.push({
        oscillators: [osc1, osc2],
        gain: noteGain
      });
    }

    this.channels.set(channel, {
      chordName,
      voices
    });
  }

  stopChannel(channel, releaseSeconds = 0.28) {
    if (!this.audioCtx) return;

    const active = this.channels.get(channel);
    if (!active) return;

    this.channels.delete(channel);

    const now = this.audioCtx.currentTime;

    for (const voice of active.voices) {
      const g = voice.gain.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(Math.max(g.value, 0.0001), now);
      g.exponentialRampToValueAtTime(0.0001, now + releaseSeconds);

      for (const osc of voice.oscillators) {
        osc.stop(now + releaseSeconds + 0.04);
      }
    }
  }

  stopAll() {
    for (const channel of [...this.channels.keys()]) {
      this.stopChannel(channel, 0.30);
    }
  }

  stop() {
    this.stopAll();
  }
}

const synth = new WarmSynth();

// ============================================================
// 5. 검지 펴짐 정도 + 제스처 판별
//
// 다른 손가락은 무시한다.
// 검지를 펴면 POINT, 접으면 CLOSED.
// ============================================================

function angle3(a,b,c) {
  const bax=a.x-b.x, bay=a.y-b.y, baz=(a.z||0)-(b.z||0);
  const bcx=c.x-b.x, bcy=c.y-b.y, bcz=(c.z||0)-(b.z||0);
  const dot=bax*bcx+bay*bcy+baz*bcz;
  const m1=Math.hypot(bax,bay,baz), m2=Math.hypot(bcx,bcy,bcz);
  if (!m1 || !m2) return 0;
  const cos=Math.max(-1,Math.min(1,dot/(m1*m2)));
  return Math.acos(cos)*180/Math.PI;
}

function dist3(a,b) {
  return Math.hypot(
    a.x-b.x,
    a.y-b.y,
    (a.z||0)-(b.z||0)
  );
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function norm(v, low, high) {
  return clamp01((v-low)/(high-low));
}

function indexRawScore(lm) {
  const pip = angle3(lm[5], lm[6], lm[7]);
  const dip = angle3(lm[6], lm[7], lm[8]);

  const palmWidth = Math.max(dist3(lm[5], lm[17]), 0.001);
  const tipDistance = dist3(lm[5], lm[8]) / palmWidth;

  const pipScore = norm(pip, 70, 175);
  const dipScore = norm(dip, 65, 175);
  const lengthScore = norm(tipDistance, 0.65, 1.60);

  return clamp01(
    pipScore * 0.42 +
    dipScore * 0.42 +
    lengthScore * 0.16
  );
}

function calibratedIndexPercent(lm, side) {
  const raw = indexRawScore(lm);

  const cal = settings.calibration?.[side] || {
    closed: 0.28,
    open: 0.86
  };

  let low = Number(cal.closed);
  let high = Number(cal.open);

  if (high < low) [low, high] = [high, low];

  if (Math.abs(high - low) < 0.05) {
    low = 0.28;
    high = 0.86;
  }

  return Math.round(
    clamp01((raw - low) / (high - low)) * 100
  );
}

function classifyGesture(lm, side) {
  const open = calibratedIndexPercent(lm, side);

  // 보정 후 기준
  if (open >= 70) return "POINT";
  if (open <= 30) return "CLOSED";
  return "OTHER";
}

// ============================================================
// 6. 화면 좌표 / 원형 메뉴
// ============================================================

function resizeCanvas() {
  canvas.width=window.innerWidth;
  canvas.height=window.innerHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function isMirroredCamera() {
  return settings.cameraFacing !== "environment";
}

function applyCameraMirror() {
  video.style.transform = isMirroredCamera() ? "scaleX(-1)" : "none";
}

function toScreenPoint(lm) {
  const x = isMirroredCamera() ? (1-lm.x) : lm.x;
  return { x:x*canvas.width, y:lm.y*canvas.height };
}

function wheelGeom(kind) {
  const w = canvas.width;
  const h = canvas.height;
  const minSide = Math.min(w,h);
  const portrait = h > w * 1.08;

  if (settings.mode === "basic") {
    if (portrait) {
      // 세로형 스마트폰: 기존보다 살짝 위로
      const outer = Math.min(w * 0.34, h * 0.18);
      const inner = outer * 0.27;

      if (kind === "left") {
        return {
          cx: w * 0.50,
          cy: h * 0.35,
          outer,
          inner
        };
      }

      return {
        cx: w * 0.50,
        cy: h * 0.67,
        outer,
        inner
      };
    }

    // PC / 가로형 폰: 좌우 배치 + 살짝 위로
    const outer = minSide * 0.22;
    const inner = minSide * 0.058;

    if (kind==="left") return {
      cx:w*0.30, cy:h*0.54,
      outer, inner
    };

    return {
      cx:w*0.70, cy:h*0.54,
      outer, inner
    };
  }

  // 사용자 코드 모드도 살짝 위로
  if (portrait) {
    const outer = Math.min(w * 0.39, h * 0.22);
    return {
      cx:w*0.50, cy:h*0.54,
      outer,
      inner:outer*0.24
    };
  }

  return {
    cx:w*0.50, cy:h*0.54,
    outer:minSide*0.27, inner:minSide*0.065
  };
}

function sectorFromPoint(p,g,count) {
  const dx=p.x-g.cx, dy=p.y-g.cy;
  const d=Math.hypot(dx,dy);
  if (d<g.inner || d>g.outer) return -1;

  let angle=Math.atan2(dy,dx)+Math.PI/2;
  if (angle<0) angle+=Math.PI*2;

  return Math.floor(angle/(Math.PI*2/count));
}

function drawWheel(labels,g,selected,pointer,centerText) {
  const n=labels.length, step=Math.PI*2/n;
  ctx.save();

  for (let i=0;i<n;i++) {
    const start=-Math.PI/2+i*step, end=start+step;

    ctx.beginPath();
    ctx.arc(g.cx,g.cy,g.outer,start,end);
    ctx.arc(g.cx,g.cy,g.inner,end,start,true);
    ctx.closePath();

    ctx.fillStyle=i===selected
      ? "rgba(255,255,255,.38)"
      : "rgba(20,25,34,.40)";
    ctx.strokeStyle="rgba(255,255,255,.42)";
    ctx.lineWidth=2;
    ctx.fill();
    ctx.stroke();

    const mid=start+step/2, r=(g.inner+g.outer)/2;
    const tx=g.cx+Math.cos(mid)*r;
    const ty=g.cy+Math.sin(mid)*r;

    ctx.fillStyle="white";
    const fontSize=Math.max(9,Math.min(20,180/n+7));
    ctx.font=`700 ${fontSize}px system-ui`;
    ctx.textAlign="center";
    ctx.textBaseline="middle";
    ctx.fillText(labels[i],tx,ty);
  }

  ctx.beginPath();
  ctx.arc(g.cx,g.cy,g.inner-2,0,Math.PI*2);
  ctx.fillStyle="rgba(0,0,0,.58)";
  ctx.fill();

  // 중앙은 텍스트 없는 비활성 영역(dead zone)으로 둔다.

  if (pointer) {
    ctx.beginPath();
    ctx.arc(pointer.x,pointer.y,10,0,Math.PI*2);
    ctx.fillStyle="rgba(255,255,255,.94)";
    ctx.fill();
  }

  ctx.restore();
}


function drawManualDualSelection(labels,g,leftIndex,rightIndex,leftPointer,rightPointer) {
  const n=labels.length;
  const step=Math.PI*2/n;

  function highlight(index) {
    if (index<0) return;

    const start=-Math.PI/2+index*step;
    const end=start+step;

    ctx.save();
    ctx.beginPath();
    ctx.arc(g.cx,g.cy,g.outer,start,end);
    ctx.arc(g.cx,g.cy,g.inner,end,start,true);
    ctx.closePath();
    ctx.fillStyle="rgba(255,255,255,.30)";
    ctx.fill();
    ctx.strokeStyle="rgba(255,255,255,.85)";
    ctx.lineWidth=3;
    ctx.stroke();
    ctx.restore();
  }

  highlight(leftIndex);

  // 같은 코드도 양손으로 누를 수 있다.
  // 같은 칸이면 하이라이트는 한 번만 보이지만 소리는 두 채널에서 독립적으로 난다.
  if (rightIndex!==leftIndex) highlight(rightIndex);

  function pointerDot(p,label) {
    if (!p) return;

    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x,p.y,11,0,Math.PI*2);
    ctx.fillStyle="rgba(255,255,255,.95)";
    ctx.fill();

    ctx.fillStyle="black";
    ctx.font="700 9px system-ui";
    ctx.textAlign="center";
    ctx.textBaseline="middle";
    ctx.fillText(label,p.x,p.y);
    ctx.restore();
  }

  pointerDot(leftPointer,"L");
  pointerDot(rightPointer,"R");

  // 중앙은 비워 둔다.
}

function drawHand(lm) {
  ctx.save();
  ctx.strokeStyle="rgba(255,255,255,.72)";
  ctx.fillStyle="rgba(255,255,255,.92)";
  ctx.lineWidth=3;

  for (const conn of HandLandmarker.HAND_CONNECTIONS) {
    const a=toScreenPoint(lm[conn.start]);
    const b=toScreenPoint(lm[conn.end]);
    ctx.beginPath();
    ctx.moveTo(a.x,a.y);
    ctx.lineTo(b.x,b.y);
    ctx.stroke();
  }

  for (let i=0;i<lm.length;i++) {
    const p=toScreenPoint(lm[i]);
    ctx.beginPath();
    ctx.arc(p.x,p.y,i===8?7:4,0,Math.PI*2);
    ctx.fill();
  }
  ctx.restore();
}

// ============================================================
// 7. 왼쪽/오른쪽 검지 펴짐 UI
// ============================================================

function setMeter(side, value) {
  const fill = side==="left" ? leftOpenFill : rightOpenFill;
  const label = side==="left" ? leftOpenValue : rightOpenValue;

  if (value == null) {
    fill.style.height="0%";
    label.textContent="--%";
    return;
  }

  fill.style.height=`${value}%`;
  label.textContent=`${value}%`;
}

function updateIndexMeters(hands) {
  setMeter("left", null);
  setMeter("right", null);

  if (!hands.length) return;

  // 사용자가 보는 화면 기준 왼쪽/오른쪽 손으로 나눈다.
  const ordered=[...hands].sort(
    (a,b)=>toScreenPoint(a[9]).x-toScreenPoint(b[9]).x
  );

  if (ordered.length>=2) {
    setMeter("left", calibratedIndexPercent(ordered[0], "left"));
    setMeter("right", calibratedIndexPercent(ordered[ordered.length-1], "right"));
    return;
  }

  const hand=ordered[0];
  const x=toScreenPoint(hand[9]).x;
  const value=calibratedIndexPercent(hand, x<canvas.width/2 ? "left" : "right");

  if (x<canvas.width/2) setMeter("left", value);
  else setMeter("right", value);
}

// ============================================================
// 8. 선택 상태
// ============================================================

const SELECT_DWELL_MS=130;
const CLOSED_DWELL_MS=150;

let leftSelected=-1, rightSelected=-1;
let leftCandidate=-1, rightCandidate=-1;
let leftCandidateSince=0, rightCandidateSince=0;
let closedSince=0;

// 사용자 코드 모드에서는 양손이 독립적인 코드를 잡는다.
const manualHands = {
  left: {
    selected: -1,
    candidate: -1,
    candidateSince: 0,
    closedSince: 0
  },
  right: {
    selected: -1,
    candidate: -1,
    candidateSince: 0,
    closedSince: 0
  }
};

function getQualities() {
  return settings.qualities.slice(0,settings.qualityCount);
}

function getManualLabels() {
  return settings.manualChords.slice(0,settings.slotCount);
}

function resetBasicSelections(stopSound=true) {
  leftSelected=rightSelected=-1;
  leftCandidate=rightCandidate=-1;

  if (stopSound) synth.stopChannel("basic");

  if (settings.mode==="basic") {
    currentChordEl.textContent="-";
  }
}

function resetManualHand(side, stopSound=true) {
  const st=manualHands[side];
  st.selected=-1;
  st.candidate=-1;
  st.candidateSince=0;
  st.closedSince=0;

  if (stopSound) synth.stopChannel(side);
  updateManualChordDisplay();
}

function resetSelections(stopSound=true) {
  resetBasicSelections(stopSound);
  resetManualHand("left",stopSound);
  resetManualHand("right",stopSound);

  if (stopSound) synth.stopAll();
  currentChordEl.textContent="-";
}

function updateManualChordDisplay() {
  const labels=getManualLabels();
  const names=[];

  if (manualHands.left.selected>=0) {
    names.push(labels[manualHands.left.selected]);
  }

  if (manualHands.right.selected>=0) {
    names.push(labels[manualHands.right.selected]);
  }

  currentChordEl.textContent=names.length ? names.join(" + ") : "-";
}

function dwellSelectBasic(idx,side,nowMs) {
  if (side==="left") {
    if (idx!==leftCandidate) {
      leftCandidate=idx;
      leftCandidateSince=nowMs;
    }

    if (idx>=0 && idx!==leftSelected &&
        nowMs-leftCandidateSince>=SELECT_DWELL_MS) {
      leftSelected=idx;
      maybePlayBasic();
    }
  }

  if (side==="right") {
    if (idx!==rightCandidate) {
      rightCandidate=idx;
      rightCandidateSince=nowMs;
    }

    if (idx>=0 && idx!==rightSelected &&
        nowMs-rightCandidateSince>=SELECT_DWELL_MS) {
      rightSelected=idx;
      maybePlayBasic();
    }
  }
}

function dwellSelectManual(idx,side,nowMs) {
  const st=manualHands[side];

  if (idx!==st.candidate) {
    st.candidate=idx;
    st.candidateSince=nowMs;
  }

  if (idx>=0 && idx!==st.selected &&
      nowMs-st.candidateSince>=SELECT_DWELL_MS) {
    st.selected=idx;

    const chord=getManualLabels()[idx];

    // 왼손과 오른손이 서로 다른 채널에서 독립적으로 울린다.
    synth.playChord(chord,side);
    updateManualChordDisplay();
  }
}

function maybePlayBasic() {
  if (leftSelected<0 || rightSelected<0) return;

  const quality=getQualities()[rightSelected];
  const chord=buildChordName(ROOTS[leftSelected],quality);

  if (!parseChord(chord)) {
    statusEl.textContent=`코드 표기를 읽을 수 없음: ${chord}`;
    return;
  }

  synth.playChord(chord,"basic");
  currentChordEl.textContent=chord;
}

function handleBasicClosed(anyClosed,nowMs) {
  if (!anyClosed) {
    closedSince=0;
    return false;
  }

  if (!closedSince) closedSince=nowMs;

  if (nowMs-closedSince>=CLOSED_DWELL_MS) {
    resetBasicSelections(true);
    statusEl.textContent="검지 접음 → 취소 / 정지";
    return true;
  }

  return false;
}

function handleManualClosed(side,isClosed,nowMs) {
  const st=manualHands[side];

  if (!isClosed) {
    st.closedSince=0;
    return false;
  }

  if (!st.closedSince) st.closedSince=nowMs;

  if (nowMs-st.closedSince>=CLOSED_DWELL_MS) {
    resetManualHand(side,true);
    return true;
  }

  return false;
}

// ============================================================
// 9. 기본 모드 / 사용자 모드
// ============================================================

function sortHandsLeftToRight(hands) {
  return [...hands].sort(
    (a,b)=>toScreenPoint(a[9]).x-toScreenPoint(b[9]).x
  );
}

function processBasic(hands,nowMs) {
  if (!hands.length) return {leftPointer:null,rightPointer:null};

  const orderedForClosed=sortHandsLeftToRight(hands);
  const anyClosed=orderedForClosed.some((h,i)=>{
    const side =
      orderedForClosed.length>=2
        ? (i===0 ? "left" : "right")
        : (toScreenPoint(h[9]).x<canvas.width/2 ? "left" : "right");
    return classifyGesture(h,side)==="CLOSED";
  });
  if (handleBasicClosed(anyClosed,nowMs)) {
    return {leftPointer:null,rightPointer:null};
  }

  const ordered=sortHandsLeftToRight(hands);
  const leftGeom=wheelGeom("left");
  const rightGeom=wheelGeom("right");

  let leftHand=null, rightHand=null;

  if (ordered.length>=2) {
    leftHand=ordered[0];
    rightHand=ordered[ordered.length-1];
  } else {
    const only=ordered[0];
    const p=toScreenPoint(only[9]);

    const dl=Math.hypot(p.x-leftGeom.cx,p.y-leftGeom.cy);
    const dr=Math.hypot(p.x-rightGeom.cx,p.y-rightGeom.cy);

    if (dl<dr) leftHand=only;
    else rightHand=only;
  }

  let leftPointer=null, rightPointer=null;

  if (!leftHand && leftSelected >= 0) {
    leftSelected = -1;
    leftCandidate = -1;
    synth.stopChannel("basic");
    currentChordEl.textContent = "-";
  }

  if (!rightHand && rightSelected >= 0) {
    rightSelected = -1;
    rightCandidate = -1;
    synth.stopChannel("basic");
    currentChordEl.textContent = "-";
  }

  if (leftHand) {
    leftPointer=toScreenPoint(leftHand[8]);

    if (classifyGesture(leftHand,"left")==="POINT") {
      const idx=sectorFromPoint(leftPointer,leftGeom,ROOTS.length);

      if (idx < 0) {
        leftSelected = -1;
        leftCandidate = -1;
        synth.stopChannel("basic");
        currentChordEl.textContent = "-";
      } else {
        dwellSelectBasic(idx,"left",nowMs);
      }
    }
  }

  if (rightHand) {
    rightPointer=toScreenPoint(rightHand[8]);

    if (classifyGesture(rightHand,"right")==="POINT") {
      const qs=getQualities();
      const idx=sectorFromPoint(rightPointer,rightGeom,qs.length);

      if (idx < 0) {
        rightSelected = -1;
        rightCandidate = -1;
        synth.stopChannel("basic");
        currentChordEl.textContent = "-";
      } else {
        dwellSelectBasic(idx,"right",nowMs);
      }
    }
  }

  const lt="";
  const qs=getQualities();
  const rt="";

  statusEl.textContent=`왼쪽: ${lt} · 오른쪽: ${rt}`;

  return {leftPointer,rightPointer};
}

function processManual(hands,nowMs) {
  const geom=wheelGeom("manual");
  const labels=getManualLabels();

  let leftPointer=null;
  let rightPointer=null;

  if (!hands.length) {
    if (manualHands.left.selected >= 0) resetManualHand("left",true);
    if (manualHands.right.selected >= 0) resetManualHand("right",true);
    return {leftPointer,rightPointer};
  }

  const ordered=sortHandsLeftToRight(hands);

  // 화면 기준 왼쪽/오른쪽 손을 독립 채널로 사용
  let leftHand=null;
  let rightHand=null;

  if (ordered.length>=2) {
    leftHand=ordered[0];
    rightHand=ordered[ordered.length-1];
  } else {
    const hand=ordered[0];
    const x=toScreenPoint(hand[9]).x;

    if (x<canvas.width/2) leftHand=hand;
    else rightHand=hand;
  }

  // 활성화돼 있던 손이 화면에서 사라지면 그 손의 음만 정지
  if (!leftHand && manualHands.left.selected >= 0) {
    resetManualHand("left",true);
  }
  if (!rightHand && manualHands.right.selected >= 0) {
    resetManualHand("right",true);
  }

  function processOne(hand,side) {
    if (!hand) return null;

    const pointer=toScreenPoint(hand[8]);
    const gesture=classifyGesture(hand,side);

    if (handleManualClosed(side,gesture==="CLOSED",nowMs)) {
      return pointer;
    }

    if (gesture==="POINT") {
      const idx=sectorFromPoint(pointer,geom,labels.length);

      if (idx < 0) {
        resetManualHand(side,true);
      } else {
        dwellSelectManual(idx,side,nowMs);
      }
    }

    return pointer;
  }

  leftPointer=processOne(leftHand,"left");
  rightPointer=processOne(rightHand,"right");

  const leftName=
    manualHands.left.selected>=0
      ? labels[manualHands.left.selected]
      : "-";

  const rightName=
    manualHands.right.selected>=0
      ? labels[manualHands.right.selected]
      : "-";

  statusEl.textContent=
    `왼손: ${leftName} · 오른손: ${rightName}`;

  return {leftPointer,rightPointer};
}


// ============================================================
// 검지 보정
// ============================================================

let latestHandsBySide = { left:null, right:null };
let calibrationJob = null;

function updateLatestHandsBySide(hands) {
  latestHandsBySide.left = null;
  latestHandsBySide.right = null;

  if (!hands.length) return;

  const ordered = sortHandsLeftToRight(hands);

  if (ordered.length >= 2) {
    latestHandsBySide.left = ordered[0];
    latestHandsBySide.right = ordered[ordered.length - 1];
    return;
  }

  const hand = ordered[0];
  const x = toScreenPoint(hand[9]).x;

  if (x < canvas.width/2) latestHandsBySide.left = hand;
  else latestHandsBySide.right = hand;
}

function refreshCalibrationUI() {
  leftClosedSaved.textContent =
    Number(settings.calibration.left.closed).toFixed(3);
  leftOpenSaved.textContent =
    Number(settings.calibration.left.open).toFixed(3);
  rightClosedSaved.textContent =
    Number(settings.calibration.right.closed).toFixed(3);
  rightOpenSaved.textContent =
    Number(settings.calibration.right.open).toFixed(3);
}

function startCalibration(side,type) {
  if (!latestHandsBySide[side]) {
    calibrationStatus.textContent =
      `${side==="left" ? "왼쪽" : "오른쪽"} 손이 화면에 보이지 않습니다.`;
    return;
  }

  calibrationJob = {
    side,
    type,
    start: performance.now(),
    samples: []
  };

  calibrationStatus.textContent =
    `${side==="left" ? "왼쪽" : "오른쪽"} 검지를 ` +
    `${type==="closed" ? "완전히 접은 채" : "완전히 편 채"} 유지하세요...`;
}

function updateCalibration(nowMs) {
  if (!calibrationJob) return;

  const hand = latestHandsBySide[calibrationJob.side];

  if (hand) {
    calibrationJob.samples.push(indexRawScore(hand));
  }

  const elapsed = nowMs - calibrationJob.start;

  if (elapsed < 800) {
    calibrationStatus.textContent =
      `${calibrationJob.side==="left" ? "왼쪽" : "오른쪽"} ` +
      `${calibrationJob.type==="closed" ? "접힘" : "펴짐"} 보정 중...`;
    return;
  }

  const samples = calibrationJob.samples;

  if (samples.length < 5) {
    calibrationStatus.textContent =
      "보정 실패: 손이 충분히 인식되지 않았습니다.";
    calibrationJob = null;
    return;
  }

  const sorted = [...samples].sort((a,b)=>a-b);
  const lo = Math.floor(sorted.length*0.2);
  const hi = Math.ceil(sorted.length*0.8);
  const trimmed = sorted.slice(lo,hi);

  const average =
    trimmed.reduce((sum,v)=>sum+v,0) / trimmed.length;

  settings.calibration[calibrationJob.side][calibrationJob.type] = average;
  persistSettings();
  refreshCalibrationUI();

  calibrationStatus.textContent =
    `${calibrationJob.side==="left" ? "왼쪽" : "오른쪽"} ` +
    `${calibrationJob.type==="closed" ? "완전 접힘" : "완전 펴짐"} 저장 완료`;

  calibrationJob = null;
}

// ============================================================
// 10. 카메라 / 메인 루프
// ============================================================

async function openCameraStream() {
  if (activeStream) {
    for (const track of activeStream.getTracks()) {
      track.stop();
    }
    activeStream = null;
  }

  statusEl.textContent="카메라 권한 요청 중...";

  const mobile = isMobileDevice();
  let stream;

  // 모바일에서는 1280x720을 계속 AI에 넣으면 기기에 따라 너무 무거울 수 있음.
  // 640x480 정도로 낮추면 손 추적이 훨씬 안정적인 경우가 많다.
  const videoConstraints = mobile
    ? {
        width:{ideal:640},
        height:{ideal:480},
        frameRate:{ideal:30, max:30},
        facingMode:{ideal:settings.cameraFacing}
      }
    : {
        width:{ideal:1280},
        height:{ideal:720},
        frameRate:{ideal:30, max:60},
        facingMode:{ideal:settings.cameraFacing}
      };

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video:videoConstraints,
      audio:false
    });
  } catch (firstError) {
    console.warn("Preferred camera constraints failed:", firstError);

    stream = await navigator.mediaDevices.getUserMedia({
      video:{
        facingMode:{ideal:settings.cameraFacing}
      },
      audio:false
    });
  }

  activeStream = stream;
  video.srcObject = stream;
  applyCameraMirror();

  await video.play();

  lastVideoTime = -1;
  cachedHands = [];
  detectErrorCount = 0;
}

async function startCamera() {
  await openCameraStream();

  await synth.init();
  synth.setTone(settings.tone);

  statusEl.textContent="손 인식 모델 불러오는 중...";

  if (!handLandmarker) await createHandLandmarker();

  cameraStarted = true;
  startBtn.style.display="none";
  statusEl.textContent="준비 완료";

  requestAnimationFrame(loop);
}

async function restartCameraForFacingChange() {
  if (!cameraStarted) {
    applyCameraMirror();
    return;
  }

  try {
    statusEl.textContent="카메라 전환 중...";
    await openCameraStream();
    statusEl.textContent="준비 완료";
  } catch (err) {
    console.error(err);
    statusEl.textContent="카메라 전환 실패";
  }
}

function loop(nowMs) {
  if (!handLandmarker || video.readyState<2) {
    requestAnimationFrame(loop);
    return;
  }

  let hands = cachedHands;

  // 모바일 CPU에서 매 화면 refresh(60~120Hz)마다 추론하면 너무 무거움.
  // 최대 약 30fps, 그리고 실제 새 카메라 프레임이 들어왔을 때만 추론한다.
  const inferenceInterval = isMobileDevice() ? 33 : 16;
  const hasNewFrame = video.currentTime !== lastVideoTime;

  if (hasNewFrame && nowMs - lastDetectAt >= inferenceInterval) {
    try {
      const result = handLandmarker.detectForVideo(video, nowMs);
      hands = result.landmarks || [];
      cachedHands = hands;

      lastVideoTime = video.currentTime;
      lastDetectAt = nowMs;
      detectErrorCount = 0;

      if (hands.length > 0) {
        engineStatusEl.textContent = `손 인식: ${handEngineName} · ${hands.length}손`;
      } else {
        engineStatusEl.textContent = `손 인식: ${handEngineName} · 찾는 중`;
      }
    } catch (err) {
      detectErrorCount++;
      console.error("Hand detection error:", err);

      engineStatusEl.textContent =
        `손 인식 오류 ${detectErrorCount}`;

      // 화면 자체는 계속 살아 있도록 이전 결과 대신 빈 손으로 처리
      hands = [];
      cachedHands = [];
    }
  }

  updateLatestHandsBySide(hands);
  updateCalibration(nowMs);

  leftRawValue.textContent = latestHandsBySide.left
    ? indexRawScore(latestHandsBySide.left).toFixed(3)
    : "--";

  rightRawValue.textContent = latestHandsBySide.right
    ? indexRawScore(latestHandsBySide.right).toFixed(3)
    : "--";

  ctx.clearRect(0,0,canvas.width,canvas.height);

  for (const hand of hands) drawHand(hand);

  updateIndexMeters(hands);

  if (settings.mode==="basic") {
    const out=processBasic(hands,nowMs);

    const lg=wheelGeom("left");
    const rg=wheelGeom("right");
    const qs=getQualities();

    drawWheel(
      ROOTS,
      lg,
      leftSelected,
      out.leftPointer,
      ""
    );

    drawWheel(
      qs,
      rg,
      rightSelected,
      out.rightPointer,
      ""
    );
  } else {
    const out=processManual(hands,nowMs);
    const labels=getManualLabels();
    const g=wheelGeom("manual");

    drawWheel(
      labels,
      g,
      -1,
      null,
      ""
    );

    drawManualDualSelection(
      labels,
      g,
      manualHands.left.selected,
      manualHands.right.selected,
      out.leftPointer,
      out.rightPointer
    );
  }

  requestAnimationFrame(loop);
}

// ============================================================
// 11. 셋업 UI
// ============================================================

function fillCountSelect(select,min,max) {
  select.innerHTML="";
  for (let n=min;n<=max;n++) {
    const opt=document.createElement("option");
    opt.value=n;
    opt.textContent=`${n}개`;
    select.appendChild(opt);
  }
}

function ensureLength(array,count,fallbacks) {
  while (array.length<count) {
    array.push(fallbacks[array.length % fallbacks.length]);
  }
}

function buildInputGrid(container,count,values,labelSuffix="번") {
  const old=[...container.querySelectorAll("input")].map(x=>x.value.trim());
  container.innerHTML="";

  for (let i=0;i<count;i++) {
    const row=document.createElement("label");
    row.className="manualRow";

    const label=document.createElement("span");
    label.textContent=`${i+1}${labelSuffix}`;

    const input=document.createElement("input");
    input.dataset.index=i;
    input.value=old[i] ?? values[i] ?? "";

    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  }
}

function buildQualityInputs() {
  const count=Number(qualityCountSelect.value);
  ensureLength(settings.qualities,count,DEFAULT_QUALITIES);
  buildInputGrid(qualityInputs,count,settings.qualities);
}

function buildManualInputs() {
  const count=Number(slotCountSelect.value);
  ensureLength(settings.manualChords,count,DEFAULT_SETTINGS.manualChords);
  buildInputGrid(manualInputs,count,settings.manualChords);
}

function updateModeSettingsUI() {
  const mode=modeSelect.value;
  basicSettings.classList.toggle("hidden",mode!=="basic");
  manualSettings.classList.toggle("hidden",mode!=="manual");
}

function openSettings() {
  modeSelect.value=settings.mode;
  cameraFacingSelect.value=settings.cameraFacing || "user";
  qualityCountSelect.value=settings.qualityCount;
  slotCountSelect.value=settings.slotCount;
  toneSlider.value=settings.tone;

  updateModeSettingsUI();
  buildQualityInputs();
  buildManualInputs();
  refreshCalibrationUI();

  settingsPanel.classList.remove("hidden");
}

function closeSettings() {
  settingsPanel.classList.add("hidden");
}

async function saveSettings() {
  const previousFacing = settings.cameraFacing || "user";
  const nextMode=modeSelect.value;
  const nextFacing=cameraFacingSelect.value;
  const nextQualityCount=Number(qualityCountSelect.value);
  const nextSlotCount=Number(slotCountSelect.value);
  const nextTone=Number(toneSlider.value);

  const nextQualities=[...qualityInputs.querySelectorAll("input")]
    .map(x=>x.value.trim());

  // 기본 모드 타입은 "C + suffix"가 유효한지 검사
  for (let i=0;i<nextQualities.length;i++) {
    const q=nextQualities[i];
    const testChord=q==="maj" ? "C" : "C"+q;

    if (!parseChord(testChord)) {
      alert(
        `오른쪽 ${i+1}번 타입 "${q}"를 읽을 수 없습니다.\n` +
        `예: maj, m, 7, add9, 7b9, maj7#11, m7b5`
      );
      return;
    }
  }

  const nextManual=[...manualInputs.querySelectorAll("input")]
    .map(x=>x.value.trim());

  for (let i=0;i<nextManual.length;i++) {
    if (!parseChord(nextManual[i])) {
      alert(
        `${i+1}번 코드 "${nextManual[i]}"를 읽을 수 없습니다.\n\n` +
        `예: Cadd9, A9, E6, Cmaj7#11, F#m7b5, G7b9, C6/9, C/E`
      );
      return;
    }
  }

  settings.mode=nextMode;
  settings.cameraFacing=nextFacing;
  settings.qualityCount=nextQualityCount;
  settings.slotCount=nextSlotCount;
  settings.tone=nextTone;

  ensureLength(settings.qualities,nextQualityCount,DEFAULT_QUALITIES);
  ensureLength(settings.manualChords,nextSlotCount,DEFAULT_SETTINGS.manualChords);

  for (let i=0;i<nextQualities.length;i++) settings.qualities[i]=nextQualities[i];
  for (let i=0;i<nextManual.length;i++) settings.manualChords[i]=nextManual[i];

  persistSettings();
  synth.setTone(settings.tone);
  resetSelections(true);
  closeSettings();

  if (previousFacing !== settings.cameraFacing) {
    await restartCameraForFacingChange();
  }

  statusEl.textContent=
    settings.mode==="basic"
      ? "기본 모드: 왼쪽 ROOT + 오른쪽 TYPE"
      : `사용자 코드 모드: ${settings.slotCount}개`;
}

function resetSettings() {
  settings=JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
  persistSettings();

  synth.setTone(settings.tone);
  resetSelections(true);

  modeSelect.value=settings.mode;
  cameraFacingSelect.value=settings.cameraFacing || "user";
  qualityCountSelect.value=settings.qualityCount;
  slotCountSelect.value=settings.slotCount;
  toneSlider.value=settings.tone;

  updateModeSettingsUI();
  buildQualityInputs();
  buildManualInputs();

  statusEl.textContent="기본값으로 되돌렸습니다.";
}

// ============================================================
// 12. 이벤트 / 초기화
// ============================================================

startBtn.addEventListener("click",()=>{
  startCamera().catch(err=>{
    console.error(err);
    statusEl.textContent=`실행 실패: ${err.name||""} ${err.message||err}`;
  });
});

settingsBtn.addEventListener("click",openSettings);
closeSettingsBtn.addEventListener("click",closeSettings);
saveSettingsBtn.addEventListener("click",()=>{
  saveSettings().catch(err=>{
    console.error(err);
    statusEl.textContent="셋업 저장 중 오류";
  });
});
resetSettingsBtn.addEventListener("click",resetSettings);

modeSelect.addEventListener("change",updateModeSettingsUI);
qualityCountSelect.addEventListener("change",buildQualityInputs);
slotCountSelect.addEventListener("change",buildManualInputs);

toneSlider.addEventListener("input",()=>{
  synth.setTone(Number(toneSlider.value));
});

leftClosedCalBtn.addEventListener("click",()=>startCalibration("left","closed"));
leftOpenCalBtn.addEventListener("click",()=>startCalibration("left","open"));
rightClosedCalBtn.addEventListener("click",()=>startCalibration("right","closed"));
rightOpenCalBtn.addEventListener("click",()=>startCalibration("right","open"));

fillCountSelect(qualityCountSelect,3,24);
fillCountSelect(slotCountSelect,3,24);

modeSelect.value=settings.mode;
cameraFacingSelect.value=settings.cameraFacing || "user";
qualityCountSelect.value=settings.qualityCount;
slotCountSelect.value=settings.slotCount;
toneSlider.value=settings.tone;

updateModeSettingsUI();
buildQualityInputs();
buildManualInputs();
refreshCalibrationUI();

applyCameraMirror();
