/* eslint-disable no-unused-vars */
import React from "react";
import avatar from "../../images/avatar.png";
import { useState, useEffect } from "react";

const Home = () => {
  const [projectName, setProjectName] = useState("");

  const [inputval, setInputval] = useState("");
  const [visible, setVisible] = useState(true);
  const handleProjectName = () => {
    setProjectName(inputval);

    setVisible(false);
  };

  useEffect(() => {
    // Load CSS file
    const styleSheet = document.createElement("link");
    styleSheet.href = "./home.css";
    styleSheet.rel = "stylesheet";
    styleSheet.type = "text/css";
    document.head.appendChild(styleSheet);

    return () => {
      // Unload CSS file
      document.head.removeChild(styleSheet);
    };
  }, []);

  return (
    <div className="h-screen bg-[#EFE2BA] p-3">
      {/* Navbar Menu  Starts Here*/}
      <div className="bg-[#d79922] h-36 border shadow-xl shadow-yellow-700 border-black rounded-3xl mb-4 p-2 flex flex-row gap-4">
        <img
          src={avatar}
          className="rounded-full w-20 h-20 ml-2 mt-3 transition duration-300 ease-in-out hover:scale-110  "
        />
        <div className="  mt-3 flex flex-col ">
          {" "}
          <div className="text-white text-3xl font-bold">Welcome User</div>
          <div>
            <button className="text-white bg-slate-600 h-10 ">Logout</button>
          </div>
        </div>
      </div>
      {/* Navbar Menu ends here */}
      {/* content starts here */}
      <div className="p-8 bg-[#eae78c]  border border-black  shadow-md rounded-3xl  black flex flex-col ">
        {/* Input Project Name */}
        {visible && (
          <div className="flex flex-row items-center justify-center">
            <div className="group">
              <input
                required=""
                type="text"
                className="input"
                value={inputval}
                onChange={(e) => setInputval(e.target.value)}
              />
              <span className="highlight"></span>
              <span className="bar"></span>
              <label>Input Project Name</label>
            </div>
            <button
              className="  bg-[#99ced3] ml-9 h-8 rounded-lg hover hover:bg-violet-600"
              onClick={handleProjectName}
            >
              Submit
            </button>
          </div>
        )}

        {/* Input Project Name Ends here */}
        {/* main content starts here  */}
        {!visible && (
          <div>
            <h1>Hello </h1>
            <h1>{projectName}</h1>
          </div>
        )}
        {/* main content ends here  */}
      </div>
      {/* content ends here  */}
    </div>
  );
};

export default Home;
