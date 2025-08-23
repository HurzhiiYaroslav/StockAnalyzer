import './App.css';
import Sidebar from './components/Sidebar/Sidebar';
import StockDashboard from './pages/StockDashboard'; 

function App() {
  return (
    <div className="app-container">
      
      <Sidebar />
      <div className="main-content">
        <StockDashboard /> 
      </div>
    </div>
  );
}

export default App;