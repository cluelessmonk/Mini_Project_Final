/* eslint-disable no-unused-vars */
import React from "react";
import { useState, useEffect } from "react";
import axios from "axios";
const InsertImage = () => {
  const [numComponents, setNumComponents] = useState(0);
  const [temp, setTemp] = useState(0);
  const [disable, setDisable] = useState(false);
  const [files, setFiles] = useState("");

  const handleChange = () => {
    const number = parseInt(temp);
    if (!isNaN(number) && number >= 2) {
      setNumComponents(number);
      setDisable(true);
    } else {
      alert("Invalid number try again");
    }
  };

  //css addition
  useEffect(() => {
    // Load CSS file
    const styleSheet = document.createElement("link");
    styleSheet.href = "src/components/InsertImage/insertimage.css";
    styleSheet.rel = "stylesheet";
    styleSheet.type = "text/css";
    document.head.appendChild(styleSheet);

    return () => {
      // Unload CSS file
      document.head.removeChild(styleSheet);
    };
  }, []);

  //Handle imgae Upload
  const handleUpload = async () => {
    const formData = new FormData();
    const email = "abcd@gmail.com";
    formData.append("email", email);
    // console.log();
    formData.append("folderName", temp);
    setTemp(temp - 1);
    for (const file of files) {
      formData.append("images", file);
    }
    console.log(formData);
    console.log("Helre");
    try {
      await axios
        .post("http://localhost:5000/upload", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((response) => {
          console.log("Response From Server: ", response.data);
        })
        .catch((err) => {
          console.log("Error From Server: ", err);
        });
    } catch (error) {
      console.error("Error uploading files:", error);
    }
  };

  //Handle change of files
  const handleFileChange = (e) => {
    setFiles(e.target.files);
  };

  //This is the premade div
  const PreMadeDiv = () => {
    // Destructuring key from props
    return (
      <div className="mt-10 flex flex-row items-center justify-center ">
        <input
          type="file"
          multiple
          onChange={(e) => {
            setFiles(e.target.files);
          }}
        />
        <button className="w-28 hover:bg-green-400" onClick={handleUpload}>
          <div className="svg-wrapper-1">
            <div className="svg-wrapper">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                width="30"
                height="30"
                className="icon"
              >
                <path d="M22,15.04C22,17.23 20.24,19 18.07,19H5.93C3.76,19 2,17.23 2,15.04C2,13.07 3.43,11.44 5.31,11.14C5.28,11 5.27,10.86 5.27,10.71C5.27,9.33 6.38,8.2 7.76,8.2C8.37,8.2 8.94,8.43 9.37,8.8C10.14,7.05 11.13,5.44 13.91,5.44C17.28,5.44 18.87,8.06 18.87,10.83C18.87,10.94 18.87,11.06 18.86,11.17C20.65,11.54 22,13.13 22,15.04Z"></path>
              </svg>
            </div>
          </div>
          <span>Upload</span>
        </button>
      </div>
    );
  };

  return (
    <div>
      <div className="flex flex-row items-center justify-center gap-4">
        <input
          type="number"
          value={temp}
          disabled={disable}
          onChange={(e) => {
            setTemp(e.target.value);
          }}
        />
        <button
          className="bg-red-400  flex justify-center rounded-2xl text-white font-bold hover:bg-green-400 cursor-pointer w-28 "
          onClick={handleChange}
        >
          Submit
        </button>
        <h1>Add Number of Classifications</h1>
      </div>
      <div className="mt-11 flex items-center justify-center text-white font-bold underline text-3xl">
        <h1>Upload Images for each classsifications</h1>
      </div>

      <div>
        {[...Array(numComponents)].map((_, index) => (
          <PreMadeDiv key={index} />
        ))}
      </div>
    </div>
  );
};

export default InsertImage;
