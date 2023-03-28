const express = require("express");
const path = require("path");
const bcrypt = require("bcryptjs");
const spawn = require("child_process").spawn;
const moment = require("moment");
require("./db/conn");
const Admin = require("./models/adminRegDb");
const Student = require("./models/studDetailsDb");
const { mlSchema, MLstatus } = require("./models/ML");
const IOTstatus = require("./models/IOT");
const CNstatus = require("./models/CN");

const app = express();
const port = process.env.PORT || 3000;
const static_path = path.join(__dirname, "../public");

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use(express.static(static_path));
app.set("view engine", "hbs");

app.get("/", (req, res) => {
  res.render("front");
});

app.get("/home", (req, res) => {
  res.render("home");
});

app.get("/login", (req, res) => {
  res.render("home", { check: "checked" });
});

app.get("/register", (req, res) => {
  res.render("adminReg");
});

app.post("/register", async (req, res) => {
  try {
    const newAdmin = new Admin({
      username: req.body.username,
      email: req.body.email,
      subject: req.body.subject,
      password: req.body.password,
    });
    const registered = await newAdmin.save();
    res.status(201).render("home", {
      check: "checked",
      msg: "\u2713 You have successfully registered! Please login.",
    });
  } catch (err) {
    res.status(400).send(err);
  }
});

var loginAdminName;
var loginAdminSubject;
app.post("/login", async (req, res) => {
  try {
    const username = req.body.username;
    const password = req.body.password;

    loginAdminName = username;

    const admin = await Admin.findOne({ username });

    loginAdminSubject = admin.subject;

    const isPassMatched = await bcrypt.compare(password, admin.password);

    if (isPassMatched) {
      res.status(201).render("admin", { name: username });
    } else {
      res.render("home", {
        check: "checked",
        error: "\u274C Invalid login details",
      });
    }
  } catch (err) {
    res.status(400).render("home", {
      check: "checked",
      error: "\u274C Invalid login details",
    });
  }
});

let result = {};
let curDate = moment(new Date()).format("DD/MM/YYYY");
console.log(curDate);
app.post("/startAttendance", async (req, res) => {
  try {
    if (loginAdminSubject === "ml") {
      result = await MLstatus.findOne({ Date: curDate });
    } else if (loginAdminSubject === "iot") {
      result = await IOTstatus.findOne({ Date: curDate });
    } else if (loginAdminSubject === "cn") {
      result = await CNstatus.findOne({ Date: curDate });
    }
  } catch (error) {
    console.log(error);
  }
  console.log(result);
  if (result == null) {
    const process = spawn("python", ["src/recognize.py", loginAdminSubject]);

    process.stdout.on("data", (data) => {
      console.log(data.toString());
    });

    process.stderr.on("data", (data) => {
      console.error(`stderr: ${data}`);
    });

    res.render("admin", { name: loginAdminName });
  } else {
    res.render("admin", {
      name: loginAdminName,
      msg: "You have already taken the attendance.",
    });
  }
});

app.get("/student", (req, res) => {
  res.render("student", { hide: "hide" });
});

app.get("/details", (req, res) => {
  res.render("details");
});

var currId, studName;
app.post("/details", async (req, res) => {
  try {
    const newStudent = new Student({
      id: req.body.id,
      name: req.body.name,
      course: req.body.course,
      semester: req.body.semester,
    });
    const admitted = await newStudent.save();

    res.status(201).render("student", {
      name: req.body.name,
      msg: "\u2713 You have successfully admitted!",
    });
    currId = req.body.id;
    studName = req.body.name;
  } catch (err) {
    res.status(400).send(err);
  }
});

app.post("/student", (req, res) => {
  const process = spawn("python", ["src/generate.py", currId]);

  process.stdout.on("data", (data) => {
    console.log(data.toString());
  });

  process.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  res.render("student", { name: studName, hide: "hide" });
});

app.get("/attendance", async (req, res) => {
  try {
    let loginAdminCollection;

    if (loginAdminSubject === "ml") {
      loginAdminCollection = MLstatus;
    } else if (loginAdminSubject === "iot") {
      loginAdminCollection = IOTstatus;
    } else {
      loginAdminCollection = CNstatus;
    }

    if (loginAdminCollection) {
      await loginAdminCollection
        .find({}, function (err, status) {
          res.render("adminSheet", {
            name: loginAdminName,
            statusList: status,
          });
        })
        .clone();
    } else {
      res.render("adminSheet", { name: loginAdminName });
    }
  } catch (error) {
    console.log(error);
  }
});

app.get("/proceed", (req, res) => {
  res.render("student", { check: "checked" });
});

app.post("/proceed", async (req, res) => {
  try {
    let ml, cn, iot, fields;
    const id = req.body.id;

    const stud = await Student.findOne({ id });
    if (id == 140001) {
      fields = { Date: 1, status: "$140001", _id: 0 };
    } else if (id == 140002) {
      fields = { Date: 1, status: "$140002", _id: 0 };
    } else if (id == 140003) {
      fields = { Date: 1, status: "$140003", _id: 0 };
    } else if (id == 140004) {
      fields = { Date: 1, status: "$140004", _id: 0 };
    } else if (id == 140005) {
      fields = { Date: 1, status: "$140005", _id: 0 };
    }

    if (stud != null) {
      if (MLstatus && CNstatus && IOTstatus) {
        ml = await MLstatus.aggregate([{ $project: fields }]);
        cn = await CNstatus.aggregate([{ $project: fields }]);
        iot = await IOTstatus.aggregate([{ $project: fields }]);
        res.render("studSheet", {
          name: stud.name,
          mlList: ml,
          cnList: cn,
          iotList: iot,
        });
      } else if (MLstatus && CNstatus) {
        res.render("studSheet", {
          name: stud.name,
          mlList: ml,
          cnList: cn,
        });
      } else if (MLstatus && IOTstatus) {
        res.render("studSheet", {
          name: stud.name,
          mlList: ml,
          iotList: iot,
        });
      } else if (CNstatus && IOTstatus) {
        res.render("studSheet", {
          name: stud.name,
          cnList: cn,
          iotList: iot,
        });
      } else if (MLstatus) {
        res.render("studSheet", {
          name: stud.name,
          mlList: ml,
        });
      } else if (CNstatus) {
        res.render("studSheet", {
          name: stud.name,
          cnList: cn,
        });
      } else if (IOTstatus) {
        res.render("studSheet", {
          name: stud.name,
          iotList: iot,
        });
      } else {
        res.render("studSheet", {
          name: stud.name,
        });
      }
    } else {
      res.render("student", {
        check: "checked",
        error: "\u274C Invalid roll number",
      });
    }
  } catch (err) {
    res.status(400).render("student", {
      check: "checked",
      error: "\u274C Invalid roll number",
    });
  }
});

app.listen(port, () => {
  console.log(`Server is running at localhost:${port}`);
});
