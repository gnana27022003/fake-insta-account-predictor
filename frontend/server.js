const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const path = require("path");

const app = express();
const PORT = 3000; // Change if needed

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views")); // Set views folder
app.use(express.static(path.join(__dirname, "public"))); // For CSS & JS files
app.use(express.static(path.join(__dirname, 'images')));

// Routes
app.get("/", (req, res) => {
    res.render("home"); // Load home page
});

app.get("/form", (req, res) => {
    res.render("form"); // Load form page
});

app.get("/about", (req, res) => {
    res.render("about"); // Load form page

});

app.get("/view", (req, res) => {
    res.render("view"); // Load form page
});

app.get("/insights", (req, res) => {
    res.render("insights"); // Load form page
});

app.post("/predict", async (req, res) => {
    try {
        // Send data to Flask backend
        const response = await axios.post("http://127.0.0.1:5000/predict", req.body);
        res.json(response.data); // Return prediction result to frontend
    } catch (error) {
        console.error("Error connecting to backend:", error.message);
        res.status(500).json({ error: "Server error" });
    }
});

// Start Server
app.listen(PORT, () => {
    console.log(`Frontend running at http://localhost:${PORT}`);
});
