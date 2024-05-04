import React from "react";
import FlipCard from "../components/LoginSignup/LoginSignupCard";
import bg from "../images/bg3.jpg";

const LoginPage = () => {
  return (
    <div className=" relative  min-h-screen flex flex-col items-center   ">
      <div className="fixed inset-0 flex items-center justify-center bg-gray-800 bg-opacity-50">
        <div className="absolute top-0 bg-cover w-full h-full z-[-1]">
          <img src={bg} className="w-full h-full  filter blur-lg " />
        </div>
        <div className=" rounded-lg p-60 relative">
          <div className="absolute inset-0 bg-transparent opacity-50 blur-sm"></div>
          <div className="relative z-10">
            <FlipCard />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
