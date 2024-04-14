import React, { useEffect, useState } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import { useNavigate } from "react-router-dom";

function FlipCard() {
  const navigate = useNavigate();
  const [name, setName] = useState();
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();

  useEffect(() => {
    // Load CSS file
    const styleSheet = document.createElement("link");
    styleSheet.href = "./cssloginsignup.css";
    styleSheet.rel = "stylesheet";
    styleSheet.type = "text/css";
    document.head.appendChild(styleSheet);

    return () => {
      // Unload CSS file
      document.head.removeChild(styleSheet);
    };
  }, []);

  const handleSubmit1 = (e) => {
    e.preventDefault();

    console.log(`the email is ${email}, password is ${password}`);
    axios
      .post("http://localhost:3001/login", { email, password })
      .then((result) => {
        console.log(result);
        if (result.data == "Success") {
          console.log("Successfully logged in");
          navigate("/homepage");
        }
      })
      .catch((err) => console.log(err));
  };
  const handleSubmit2 = (e) => {
    e.preventDefault();
    console.log(
      `The email is ${email}, password is ${password} the name is ${name}`
    );
    axios
      .post("http://localhost:3001/register", { name, email, password })
      .then((response) => {
        console.log(response);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  return (
    <div>
      <div className="wrapper">
        <div className="card-switch">
          <label className="switch">
            <input type="checkbox" className="toggle" />
            <span className="slider"></span>
            <span className="card-side"></span>
            <div className="flip-card__inner">
              {/* Login Card */}
              <div className="flip-card__front">
                <div className="title">Log in</div>
                <form className="flip-card__form" onSubmit={handleSubmit1}>
                  <input
                    className="flip-card__input"
                    name="email"
                    placeholder="Email"
                    type="email"
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  <input
                    className="flip-card__input"
                    name="password"
                    placeholder="Password"
                    type="password"
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <button className="flip-card__btn" type="submit">
                    Let's go!
                  </button>
                </form>
              </div>
              {/* Signup Card */}
              <div className="flip-card__back">
                <div className="title">Sign up</div>
                <form className="flip-card__form" onSubmit={handleSubmit2}>
                  <input
                    className="flip-card__input"
                    placeholder="Name"
                    type="text"
                    onChange={(e) => setName(e.target.value)}
                  />
                  <input
                    className="flip-card__input"
                    name="email"
                    placeholder="Email"
                    type="email"
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  <input
                    className="flip-card__input"
                    name="password"
                    placeholder="Password"
                    type="password"
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <button className="flip-card__btn" type="submit">
                    Confirm!
                  </button>
                </form>
              </div>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}

export default FlipCard;
