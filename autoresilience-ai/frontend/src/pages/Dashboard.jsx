import { useEffect, useState } from "react";

import API from "../services/api";

import KPISection from "../components/KPISection";
import SupplierTable from "../components/SupplierTable";
import SupplierForm from "../components/SupplierForm";
import RiskChart from "../components/RiskChart";

import RiskDonutChart from "../components/RiskDonutChart";
import RiskGauge from "../components/RiskGauge";
import AIInsights from "../components/AIInsights";

import ForecastChart
from "../components/ForecastChart";

import RecommendationPanel
from "../components/RecommendationPanel";

import ExecutiveDashboard
from "../components/ExecutiveDashboard";

function Dashboard() {


  const [suppliers, setSuppliers] = useState([]);

  const [forecast,setForecast] =
useState([]);	

  const loadSuppliers = async () => {

    const res = await API.get("/suppliers");

    setSuppliers(res.data);
  };

  useEffect(() => {

    API.get("/forecast")
.then(res=>{

 setForecast(
 res.data
 );

});

    loadSuppliers();
  }, []);

  return (
  <div className="min-h-screen text-white">

    <div className="p-6">

      <h1 className="text-5xl font-bold mb-6">
        🚗 AutoResilience AI
      </h1>

      <KPISection suppliers={suppliers} />

      <div className="grid md:grid-cols-2 gap-6 mt-6">

        <RiskDonutChart
          suppliers={suppliers}
        />

        <RiskGauge
          suppliers={suppliers}	
        />

      </div>

      <div className="grid md:grid-cols-2 gap-6 mt-6">

        <AIInsights
          suppliers={suppliers}
        />

        <SupplierForm
          refresh={loadSuppliers}
        />

      </div>

      autoresilience

      <div className="mt-6">

        <SupplierTable
          suppliers={suppliers}
        />

      </div>

    </div>

  </div>
);
}

export default Dashboard;