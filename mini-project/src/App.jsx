// import MessagePopup from "./components/OtherComponents/temp";
import LoginPage from "./pages/LoginPage";
import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="" element={<LoginPage />} />
        <Route path="/homepage" element={<LandingPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
