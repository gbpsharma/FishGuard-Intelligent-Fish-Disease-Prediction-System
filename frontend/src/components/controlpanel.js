import React from "react";

function ControlPanel({settings,setSettings})
{
	return(
		<div className="card">

			<h2>Settings</h2>

			<label>
				<input
					type="checkbox"
					checked={settings.showGradcam}
					onChange={(e)=>setSettings({...settings,showGradcam:e.target.checked})}
				/>
				Show Grad-CAM
			</label>

			<br/>

			<label>
				Confidence Threshold:
			</label>

			<input
				type="range"
				min="50"
				max="95"
				value={settings.threshold}
				onChange={(e)=>setSettings({...settings,threshold:e.target.value})}
			/>

			<p>{settings.threshold}%</p>

		</div>
	);
}

export default ControlPanel;