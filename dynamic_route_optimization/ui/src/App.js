import React, {useState} from 'react';
import axios from 'axios';
function App(){
  const [resp, setResp] = useState(null);
  async function send(){
    const sample = await fetch('/data/sample_requests.json').then(r=>r.json())
    const res = await axios.post('http://localhost:8000/optimize', sample)
    setResp(res.data)
  }
  return (
    <div style={{padding:20}}>
      <h1>Dynamic Route Optimizer - UI</h1>
      <button onClick={send}>Send Sample Optimize Request</button>
      <pre>{JSON.stringify(resp, null, 2)}</pre>
    </div>
  )
}
export default App;