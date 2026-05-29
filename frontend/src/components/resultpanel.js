import React from "react";

function ResultPanel({result})
{
	if(!result) return null;

	return(
		<div className="card result">

			<h2>Result</h2>

			<h3>{result.prediction}</h3>

			<p>Confidence: {result.confidence.toFixed(2)}%</p>

			{result.image && (
				<img
					src={`data:image/jpeg;base64,${result.image}`}
					alt="result"
				/>
			)}

		</div>
	);
}

export default ResultPanel;