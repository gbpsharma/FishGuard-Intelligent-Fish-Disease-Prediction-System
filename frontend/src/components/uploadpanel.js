import React from "react";

function UploadPanel({file,setFile})
{
	return(
		<div className="card">

			<h2>Upload Image</h2>

			<input type="file" onChange={(e)=>setFile(e.target.files[0])} />

			{file && <p>{file.name}</p>}

		</div>
	);
}

export default UploadPanel;