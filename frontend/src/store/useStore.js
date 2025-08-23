import { create } from 'zustand';

const API_BASE_URL = 'http://localhost:5000/api';

// create - це функція з zustand, яка створює хук-сховище
export const useStore = create((set, get) => ({
  // --- STATE (Стан) ---
  tickers: [],
  searchQuery: '',
  tickersLoading: false,
  tickersError: null,

  selectedTicker: null,
  predictionData: null,
  predictionLoading: false,
  predictionError: null,

  // --- ACTIONS (Дії, що змінюють стан) ---

  // Дії для тікерів
  setSearchQuery: (query) => set({ searchQuery: query }),

  fetchTickers: async () => {
    set({ tickersLoading: true, tickersError: null });
    try {
      const response = await fetch(`${API_BASE_URL}/tickers`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to fetch tickers');
      set({ tickers: data.tickers });
    } catch (err) {
      set({ tickersError: err.message });
    } finally {
      set({ tickersLoading: false });
    }
  },

  searchTickers: async (query) => {
    if (!query.trim()) {
      get().fetchTickers(); // get() дозволяє викликати інші дії зі стору
      return;
    }
    set({ tickersLoading: true, tickersError: null });
    try {
      const response = await fetch(`${API_BASE_URL}/tickers/search?query=${encodeURIComponent(query)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to search tickers');
      set({ tickers: data.tickers });
    } catch (err) {
      set({ tickersError: err.message });
    } finally {
      set({ tickersLoading: false });
    }
  },

  // Дії для прогнозу
  fetchPrediction: async (ticker) => {
    if (!ticker) return;
    set({ 
      selectedTicker: ticker, 
      predictionLoading: true, 
      predictionError: null, 
      predictionData: null 
    });

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to get prediction');
      set({ predictionData: data });
    } catch (err) {
      set({ predictionError: err.message });
    } finally {
      set({ predictionLoading: false });
    }
  },
}));