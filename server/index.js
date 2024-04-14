const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const EmployeeModel = require("./models/Employee");
const app = express();
const jwt = require("jsonwebtoken");
const cookieParser = require("cookie-parser");
app.use(express.json());
app.use(cors());
app.use(cookieParser());
app.use(express.urlencoded({ extended: false }));

mongoose.connect("mongodb://127.0.0.1:27017/employee");

app.post("/register", (req, res) => {
  EmployeeModel.create(req.body)
    .then((employees) => res.json(employees))
    .catch((err) => res.json(err));
});
const JWT_SECRET = 12908210398;
app.post("/login", (req, res) => {
  const { email, password } = req.body;
  EmployeeModel.findOne({ email: email }).then((user) => {
    if (user) {
      if (user.password == password) {
        res.json("Success");
        // jwt.sign(
        //   { email: user.email, password: user.password, name: user.name },
        //   JWT_SECRET,
        //   {},
        //   (err, token) => {
        //     if (err) {
        //       console.log(err);
        //     }

        //     res.cookie("token", token).json(user);
        //   }
        // );
      } else {
        res.json("Incorrect password");
      }
    } else {
      res.json("No User Exist with this email");
    }
  });
});

app.listen(3001, () => {
  console.log("app is listening on 3001");
});
