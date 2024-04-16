const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const cors = require("cors"); // Import cors
const axios = require("axios");
const app = express();
const bodyParser = require("body-parser");
// Parse JSON bodies
app.use(bodyParser.json());

// Parse URL-encoded bodies
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors()); // Enable CORS for all routes

// Multer Storage Configuration
const storage = multer.diskStorage({
  destination: async function (req, file, cb) {
    let folderPath = `uploads/${req.body.email}`;
    console.log("multer");
    try {
      // Check if the folder already exists
      await fs.promises.access(folderPath);
      console.log("Folder already exists.");
    } catch (err) {
      // If the folder doesn't exist, create it
      await fs.promises.mkdir(folderPath);
      console.log("Folder created successfully.");
    }

    folderPath = `uploads/${req.body.email}/${req.body.folderName}`;
    try {
      // Check if the folder already exists
      await fs.promises.access(folderPath);
    } catch (err) {
      // If the folder doesn't exist, create it
      await fs.promises.mkdir(folderPath, { recursive: true });
    }

    cb(null, folderPath);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage });

// Multer Middleware to handle file uploads
app.post("/upload", upload.array("images"), (req, res) => {
  // Handle file upload here
  const folderPath = path.join(__dirname, "uploads"); // __dirname represents the current directory
  const folderPath1 = path.join(__dirname, req.body.email); // __dirname represents the current directory
  const filePath = path.join(folderPath1, "success.txt");
  const fileContent = "This is a success message!";

  // Create the uploads folder if it doesn't exist
  if (!fs.existsSync(folderPath)) {
    fs.mkdirSync(folderPath);
  }

  // Write to the success.txt file
  fs.writeFile(filePath, fileContent, (err) => {
    if (err) {
      console.error("Error creating file:", err);
    } else {
      console.log("File created successfully:", filePath);
    }
  });
  console.log(req.body);

  res.send("Files uploaded successfully");
});
// app.post('/model', async (req, res) => {

//   console.log(req.body);
//   const email=req.body.email;
//   const noOfClasses=req.body.noOfClasses;
//   axios.post(`http://localhost:8000/run_model/${email}`,noOfClasses).then(response => {
//     console.log(response.data.accuracy);
//   }).catch(err => {
//     console.error(err);
//   });
// });

app.listen(5000, () => {
  console.log("Server is running on port 5000");
});

app.post("/register", (req, res) => {
  EmployeeModel.create(req.body)
    .then((employees) => res.json(employees))
    .catch((err) => res.json(err));
});
app.post("/login", (req, res) => {
  const { email, password } = req.body;
  EmployeeModel.findOne({ email: email }).then((user) => {
    if (user) {
      if (user.password == password) {
        res.json("Success");
      } else {
        res.json("Incorrect password");
      }
    } else {
      res.json("No User Exist with this email");
    }
  });
});
// // Login And registers middlewares ends here
// const PORT = process.env.PORT || 3001;
// app.listen(PORT, () => {
//   console.log(`Server is running on port ${PORT}`);
// });
