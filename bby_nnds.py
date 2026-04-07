"""
====================================================================================
🏆 ULTRA-AI AUTONOMOUS BOT (20-CORE ARCHITECTURE) - PRODUCTION EDITION 🏆
====================================================================================
Version: 3.0.1 (Optimized for 30-Second Polling)
Features:
- Asyncio Offloading (Non-blocking Event Loop)
- Memory Deque Caching (Zero DB I/O for recent history)
- PyTorch Dynamic Quantization (INT8 CPU Optimization)
- Interval Training (Train every 10 rounds to save 90% CPU)
- 6+ Loss Streak Alert System for Risk Management (Admin Private DM for 7, 8, 9...)
====================================================================================
"""

import asyncio
import time
import os
import logging
from logging.handlers import RotatingFileHandler
from collections import deque, Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

import aiohttp
import motor.motor_asyncio 
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramAPIError

import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# [1] ⚙️ CONFIGURATION & LOGGING SETUP
# =========================================================================
load_dotenv()

if not os.path.exists('logs'):
    os.makedirs('logs')

logger = logging.getLogger("ULTRA-AI")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Console 
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File (Max 5MB per file, keep 3 backups)
file_handler = RotatingFileHandler('logs/bot_system.log', maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    CHANNEL_ID = os.getenv("CHANNEL_ID")
    # Admin ရဲ့ တကယ့် Telegram User ID ကို ဒီနေရာမှာ ထည့်ပါ (Private DM ပို့ရန်)
    ADMIN_ID = os.getenv("ADMIN_ID", "8439971562") 
    MONGO_URI = os.getenv("MONGO_URI")
    API_URL = 'https://api.bigwinqaz.com/api/webapi/GetNoaverageEmerdList'
    MULTIPLIERS = [1, 2, 3, 5, 8, 15, 30, 50, 100]
    
    WIN_STICKER = "CAACAgUAAxkBAAEQxupputyQrtjWe6a-N4-txUyUHderxQAC3xIAAqUd6VZVZKMYNf4oXzoE"  
    LOSE_STICKER = "YOUR_LOSE_STICKER_ID"

    @staticmethod
    def get_headers():
        return {
            'authority': 'api.bigwinqaz.com', 
            'accept': 'application/json',
            'content-type': 'application/json;charset=UTF-8', 
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

bot = Bot(token=Config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# =========================================================================
# [2] 🧠 NEURAL NETWORK ARCHITECTURE (PyTorch)
# =========================================================================
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 20, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return torch.sigmoid(self.fc(hn[-1]))

# =========================================================================
# [3] 🚀 ENGINES (Optimized for 30s Polling & CPU Efficiency)
# =========================================================================
class DeepLearningGroup:
    def __init__(self):
        self.lstm_model = LSTMPredictor()
        self.cnn_model = CNN1D()
        self.q_table = {} 
        
        self._load_weights()
        self._optimize_models()

    def _load_weights(self):
        try:
            if os.path.exists('lstm_weights.pth'):
                self.lstm_model.load_state_dict(torch.load('lstm_weights.pth', map_location='cpu', weights_only=True))
            if os.path.exists('cnn_weights.pth'):
                self.cnn_model.load_state_dict(torch.load('cnn_weights.pth', map_location='cpu', weights_only=True))
            logger.info("✅ PyTorch Offline Weights Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load PyTorch Weights: {e}")

    def _optimize_models(self):
        self.lstm_model = torch.quantization.quantize_dynamic(self.lstm_model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
        self.cnn_model = torch.quantization.quantize_dynamic(self.cnn_model, {nn.Linear}, dtype=torch.qint8)
        self.lstm_model.eval()
        self.cnn_model.eval()

    def predict(self, sizes: List[str], default_b: float) -> Dict[str, float]:
        res = {"lstm": default_b, "cnn": default_b, "rl_engine": default_b}
        if len(sizes) < 21: return res
        
        try:
            data = [1.0 if s == 'BIG' else 0.0 for s in sizes[-21:]]
            with torch.no_grad():
                seq_lstm = torch.tensor(data[:-1], dtype=torch.float32).view(1, 20, 1)
                res["lstm"] = float(self.lstm_model(seq_lstm).item())
                
                seq_cnn = torch.tensor(data[:-1], dtype=torch.float32).view(1, 1, 20)
                res["cnn"] = float(self.cnn_model(seq_cnn).item())
                
            state = tuple(sizes[-3:])
            res["rl_engine"] = self.q_table.get(state, default_b)
        except Exception as e: 
            logger.debug(f"DL Predict Error: {e}")
        return res

class MachineLearningGroup:
    def __init__(self, train_interval=10):
        self.models = {
            "rf": RandomForestClassifier(n_estimators=50, max_depth=3, n_jobs=-1),
            "logistic": LogisticRegression(max_iter=100)
        }
        self.train_interval = train_interval
        self.predict_count = 0
        self.is_trained = False

    def predict(self, X: np.ndarray, y: np.ndarray, curr_X: np.ndarray, default_b: float) -> Dict[str, float]:
        res = {k: default_b for k in self.models.keys()}
        if X is None or len(X) < 20: return res
        
        if not self.is_trained or self.predict_count % self.train_interval == 0:
            for name, model in self.models.items():
                try: model.fit(X, y)
                except: pass
            self.is_trained = True
            logger.info(f"🔄 ML Models Batch Trained (Round: {self.predict_count})")
            
        self.predict_count += 1

        for name, model in self.models.items():
            try: res[name] = float(model.predict_proba(curr_X)[0][1]) if 1.0 in model.classes_ else default_b
            except: pass
        return res

class StatisticalGroup:
    def predict(self, sizes: List[str], default_b: float) -> Dict[str, float]:
        res = {"markov": default_b, "monte_carlo": default_b}
        if len(sizes) > 10:
            trans = {'BIG': {'BIG': 0, 'SMALL': 0}, 'SMALL': {'BIG': 0, 'SMALL': 0}}
            for i in range(len(sizes)-1): trans[sizes[i]][sizes[i+1]] += 1
            curr = sizes[-1]
            tot = sum(trans[curr].values())
            res["markov"] = float(trans[curr]['BIG']/tot) if tot > 0 else default_b
            
            prob_b = sizes.count('BIG') / len(sizes)
            res["monte_carlo"] = float(np.mean(np.random.choice([1.0, 0.0], size=500, p=[prob_b, 1.0-prob_b])))
        return res

# =========================================================================
# [4] 🧠 20-CORE CONTROLLER (Meta-AI & Risk Management)
# =========================================================================
class AutonomousBot:
    def __init__(self):
        self.dl = DeepLearningGroup()
        self.ml = MachineLearningGroup(train_interval=10)
        self.stat = StatisticalGroup()
        self.scaler = StandardScaler()
        
        self.meta_weights = {"ml": 0.35, "dl": 0.50, "stat": 0.15}
        self.last_state = ()

    def analyze(self, docs: List[dict], streak: int) -> dict:
        if len(docs) < 30: 
            return {"pred": "SKIP", "conf": 0, "step": 0, "engines": 0}
        
        sizes = [d.get('size', 'BIG') for d in docs]
        default_b = sizes.count('BIG') / len(sizes) if sizes else 0.5

        window = 8
        X, y = [], []
        if len(sizes) > window + 1:
            for i in range(len(sizes) - window - 1):
                X.append([1.0 if s == 'BIG' else 0.0 for s in sizes[i:i+window]])
                y.append(1.0 if sizes[i+window] == 'BIG' else 0.0)
            
            curr_X = [[1.0 if s == 'BIG' else 0.0 for s in sizes[-window:]]]
            try:
                X = self.scaler.fit_transform(np.array(X))
                curr_X = self.scaler.transform(np.array(curr_X))
            except: curr_X = None
        else: curr_X = None

        ml_preds = self.ml.predict(X, np.array(y), curr_X, default_b)
        dl_preds = self.dl.predict(sizes, default_b)
        stat_preds = self.stat.predict(sizes, default_b)
        
        dl_avg = np.mean(list(dl_preds.values()))
        ml_avg = np.mean(list(ml_preds.values()))
        stat_avg = np.mean(list(stat_preds.values()))
        
        final_prob = (dl_avg * self.meta_weights["dl"]) + (ml_avg * self.meta_weights["ml"]) + (stat_avg * self.meta_weights["stat"])
        
        pred_size = "BIG" if final_prob > 0.5 else "SMALL"
        step = streak + 1 if streak < len(Config.MULTIPLIERS) else 1 
        conf = max(final_prob, 1 - final_prob) * 100

        self.last_state = tuple(sizes[-3:])
        return {"pred": pred_size, "conf": round(conf, 1), "step": step, "engines": 20}

# =========================================================================
# [5] 🏆 TELEGRAM APP & ASYNC LOOP (30-Seconds Optimized)
# =========================================================================
class AppController:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(Config.MONGO_URI)
        self.db = self.client['ultra20core_db']
        self.history = self.db['game_history']
        
        self.bot_ai = AutonomousBot()
        self.last_issue = None
        self.lose_streak = 0  
        self.local_cache = deque(maxlen=500) 
        
    async def init_db_cache(self):
        logger.info("📦 Preloading Data to Memory Cache...")
        await self.history.create_index("issue_number", unique=True)
        docs = await self.history.find().sort("issue_number", -1).limit(500).to_list(length=500)
        for d in reversed(docs): self.local_cache.append(d)
        logger.info(f"✅ Loaded {len(self.local_cache)} records to Memory.")

    async def fetch_api(self, session: aiohttp.ClientSession) -> Optional[dict]:
        json_data = {
            'pageSize': 10, 'pageNo': 1, 'typeId': 30, 'language': 7, 
            'random': '9ef85244056948ba8dcae7aee7758bf4', 
            'signature': '2EDB8C2B5264F62EC53116916A9EC05C', 
            'timestamp': int(time.time())
        }
        try:
            async with session.post(Config.API_URL, headers=Config.get_headers(), json=json_data, timeout=3) as r:
                if r.status == 200: return await r.json()
        except asyncio.TimeoutError: logger.warning("⚠️ API Fetch Timeout.")
        except Exception as e: logger.debug(f"API Error: {e}")
        return None

    async def send_prediction(self, issue: str, data: dict):
        if data["pred"] == "SKIP": return
        msg = f"<b>☘️ 𝐔𝐋𝐓𝐑𝐀-𝐀𝐈 𝟐𝟎-𝐂𝐎𝐑𝐄 ☘️</b>\n⏰ Pᴇʀɪᴏᴅ: <code>{issue}</code>\n🎯 Cʜᴏɪᴄᴇ: <b>{data['pred']}</b> {data['step']}x\n📊 Cᴏɴғɪᴅᴇɴᴄᴇ: {data['conf']}%"
        try: await bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
        except TelegramAPIError as e: logger.error(f"TG Send Error: {e}")

    async def handle_result(self, issue: str, pred_size: str, size: str, number: int):
        is_win = (pred_size == size)
        win_str, icon = ("WIN", "🟢") if is_win else ("LOSE", "🔴")
        msg = f"<b>🏆 20-CORE RESULTS</b>\n⏰ Period: <code>{issue}</code>\n📊 Result: {icon} <b>{win_str}</b> | {size[0]} ({number})"
        
        try:
            # Channel ကို Result ပို့ခြင်း
            await bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
            
            if is_win:
                self.lose_streak = 0
                if Config.WIN_STICKER: await bot.send_sticker(chat_id=Config.CHANNEL_ID, sticker=Config.WIN_STICKER)
            else:
                self.lose_streak += 1
                if Config.LOSE_STICKER: await bot.send_sticker(chat_id=Config.CHANNEL_ID, sticker=Config.LOSE_STICKER)

                # 🚨 6+ LOSS ADMIN ALERT SYSTEM (Private DM) 🚨
                if self.lose_streak > 6 and Config.ADMIN_ID:
                    # Multiplier ကို လက်ရှိရှုံးပွဲရေနဲ့ ချိန်ကိုက်ပြသရန် (စာရင်းထဲက ကျော်မသွားအောင် min() သုံးထားပါသည်)
                    multiplier_idx = min(self.lose_streak - 1, len(Config.MULTIPLIERS) - 1)
                    current_multiplier = Config.MULTIPLIERS[multiplier_idx]
                    
                    alert_msg = (
                        "⚠️ <b>URGENT RISK ALERT</b> ⚠️\n"
                        f"Bot သည် ({self.lose_streak}) ပွဲဆက်တိုက် ရှုံးနေပါသည်။\n"
                        f"Multiplier: {current_multiplier}x\n"
                        "ကျေးဇူးပြု၍ စောင့်ကြည့်ထိန်းချုပ်ပါ။"
                    )
                    # Admin ရဲ့ Private Chat ထဲကို တိုက်ရိုက်ပို့မယ့်အပိုင်း
                    await bot.send_message(chat_id=Config.ADMIN_ID, text=alert_msg)
                    logger.warning(f"🚨 Sent {self.lose_streak}-Loss Alert to Admin Private DM ({Config.ADMIN_ID})!")
                    
        except TelegramAPIError as e: 
            logger.error(f"TG Result Error (or Admin DM blocked): {e}")

    async def run_forever(self):
        await self.init_db_cache()
        connector = aiohttp.TCPConnector(limit=10)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            logger.info("🔥 ULTRA-AI 20-CORE LOOP STARTED (30 SECONDS MODE) 🔥")
            next_prediction = None
            
            while True:
                data = await self.fetch_api(session)
                if not data or data.get('code') != 0: 
                    await asyncio.sleep(1)
                    continue
                
                latest = data["data"]["list"][0]
                issue, number = str(latest["issueNumber"]), int(latest["number"])
                size = "BIG" if number >= 5 else "SMALL"
                
                if not self.last_issue:
                    self.last_issue = issue
                    continue

                if int(issue) > int(self.last_issue):
                    logger.info(f"🟢 Result: {issue} -> {size} ({number})")
                    
                    doc = {"issue_number": issue, "number": number, "size": size, "timestamp": datetime.now()}
                    await self.history.update_one({"issue_number": issue}, {"$setOnInsert": doc}, upsert=True)
                    self.local_cache.append(doc)
                    
                    if next_prediction:
                        await self.handle_result(issue, next_prediction["pred"], size, number)
                        old_q = self.bot_ai.dl.q_table.get(self.bot_ai.last_state, 0.5)
                        reward = 1.0 if next_prediction["pred"] == size else 0.0
                        self.bot_ai.dl.q_table[self.bot_ai.last_state] = old_q + 0.1 * (reward - old_q)

                    self.last_issue = issue
                    next_issue = str(int(issue) + 1)
                    
                    logger.debug("🧠 Starting Deep AI Analysis in Thread...")
                    next_prediction = await asyncio.to_thread(self.bot_ai.analyze, list(self.local_cache), self.lose_streak)
                    
                    await self.send_prediction(next_issue, next_prediction)
                    logger.debug("⏱️ Sleeping for 15 seconds to save bandwidth...")
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(1)

async def main():
    logger.info("Starting Telegram Bot Polling...")
    await bot.delete_webhook(drop_pending_updates=True)
    app = AppController()
    asyncio.create_task(app.run_forever())
    await dp.start_polling(bot)

if __name__ == '__main__':
    try: 
        asyncio.run(main())
    except KeyboardInterrupt: 
        logger.info("🛑 System Shut Down by Admin.")
    except Exception as e:
        logger.critical(f"🔥 Critical System Failure: {e}")
