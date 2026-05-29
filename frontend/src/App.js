import React,{useState} from "react";
import axios from "axios";
import "./App.css";

function App(){

	const [file,setFile]=useState(null);
	const [preview,setPreview]=useState(null);
	const [result,setResult]=useState(null);
	const [loading,setLoading]=useState(false);

	const handleUpload=(e)=>{
		const selected=e.target.files[0];

		setFile(selected);

		if(selected){
			setPreview(URL.createObjectURL(selected));
		}
	};

	const predict=async()=>{

		if(!file){
			alert("Please upload image");
			return;
		}

		setLoading(true);

		const formData=new FormData();
		formData.append("file",file);

		try{

			const res=await axios.post(
				"http://127.0.0.1:8000/predict",
				formData
			);

			if(res.data.error){
				alert(res.data.error);
				setLoading(false);
				return;
			}

			setResult(res.data);

		}catch(err){

			alert("Backend connection failed");
			console.log(err);

		}

		setLoading(false);
	};

	const isDisease=result &&
	!["Healthy Fish","NOT_FISH","No Fish Detected"]
	.includes(result.predicted_class);

	return(
		<div className="app">

			{/* NAVBAR */}
			<div className="navbar">
				<h2>
					FishGuard : Intelligent Fish Disease Prediction
				</h2>
			</div>

			{/* MAIN */}
			<div className="container">

				<p className="subtitle">
					Upload Fish Image
				</p>

				{/* UPLOAD */}
				<div className="upload-box">

					<input
						type="file"
						accept="image/*"
						onChange={handleUpload}
					/>

					<p>
						Drag & Drop or Click to Upload
					</p>

				</div>

				{/* PREVIEW */}
				{preview && (

					<div className="preview-box">

						<img
							src={preview}
							alt=""
						/>

					</div>

				)}

				<button
					className="predict-btn"
					onClick={predict}
				>
					{loading ? "Analyzing..." : "Predict"}
				</button>

				{/* RESULT */}
				{result && (

					<div className="result">

						<div className="result-header">

							<h2>
								{result.predicted_class}
							</h2>

							<div className="confidence">

								{result.confidence
									? result.confidence.toFixed(2)+"%"
									: "N/A"}

							</div>

						</div>

						{/* DISEASE */}
						{isDisease ? (

							<>

								<div className="cards">

									<div className="card-box">

										<h4>
											Uploaded Image
										</h4>

										<img
											src={`data:image/jpeg;base64,${result.image}`}
											alt=""
										/>

									</div>

									<div className="card-box">

										<h4>
											Grad-CAM
										</h4>

										<img
											src={`data:image/jpeg;base64,${result.gradcam}`}
											alt=""
										/>

									</div>

									<div className="card-box">

										<h4>
											Infected Area
										</h4>

										<img
											src={`data:image/jpeg;base64,${result.infected}`}
											alt=""
										/>

									</div>

								</div>

								{/* TOP 3 */}
								<div className="top3">

									<h3>
										Top Predictions
									</h3>

									{result.top3.map((item,index)=>(

										<div
											className="top-item"
											key={index}
										>

											<span>
												{item[0]}
											</span>

											<span>
												{item[1].toFixed(2)}%
											</span>

										</div>

									))}

								</div>

							</>

						):(

							<div className="no-disease">

								{result.predicted_class==="Healthy Fish"
									? "Healthy Fish Detected"
									: "No Fish Detected"}

							</div>

						)}

					</div>

				)}

			</div>

		</div>
	);
}

export default App;