import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [reference, setReference] = useState('');
  const [summary, setSummary] = useState('');
  const [rougeScores, setRougeScores] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        passage: text,
        reference: reference,
      });
      setSummary(response.data.summary);
      setRougeScores(response.data.rouge_scores);  // Set ROUGE scores from response
    } catch (error) {
      console.error("Error generating the summary:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ padding: '20px', maxWidth: '600px', margin: 'auto' }}>
      <h1>Text Summarization</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="passage">Enter your text here:</label><br />
        <textarea
          id="passage"
          rows="10"
          cols="60"
          value={text}
          onChange={(e) => setText(e.target.value)}
          required
          placeholder="Paste text to summarize..."
          style={{ width: '100%', padding: '10px', margin: '10px 0' }}
        ></textarea>
        
        <label htmlFor="reference">Optional: Enter a reference summary for ROUGE evaluation:</label><br />
        <textarea
          id="reference"
          rows="5"
          cols="60"
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          placeholder="Paste reference summary here for ROUGE score..."
          style={{ width: '100%', padding: '10px', margin: '10px 0' }}
        ></textarea>
        
        <button type="submit" style={{ padding: '10px 20px', fontSize: '16px' }}>Summarize</button>
      </form>

      {loading ? (
        <p>Loading...</p>
      ) : (
        summary && (
          <div style={{ marginTop: '20px' }}>
            <h2>Summary</h2>
            <p>{summary}</p>

            {rougeScores && (
              <div style={{ marginTop: '20px' }}>
                <h3>ROUGE Scores</h3>
                <p><strong>ROUGE-1:</strong> {rougeScores.rouge1.toFixed(3)}</p>
                <p><strong>ROUGE-2:</strong> {rougeScores.rouge2.toFixed(3)}</p>
                <p><strong>ROUGE-L:</strong> {rougeScores.rougeL.toFixed(3)}</p>
              </div>
            )}
          </div>
        )
      )}
    </div>
  );
}

export default App;
