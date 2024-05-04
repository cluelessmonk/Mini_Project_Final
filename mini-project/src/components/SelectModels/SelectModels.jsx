import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Modal from "react-bootstrap/Modal";
import axios from "axios";

function MyVerticallyCenteredModal1(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Modal heading
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Centered Modal</h4>
        <p>
          Cras mattis consectetur purus sit amet fermentum. Cras justo odio,
          dapibus ac facilisis in, egestas eget quam. Morbi leo risus, porta ac
          consectetur ac, vestibulum at eros.
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}
function MyVerticallyCenteredModal2(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Modal heading
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Centered Modal</h4>
        <p>
          Cras mattis consectetur purus sit amet fermentum. Cras justo odio,
          dapibus ac facilisis in, egestas eget quam. Morbi leo risus, porta ac
          consectetur ac, vestibulum at eros.
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}
function MyVerticallyCenteredModal3(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Modal heading
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Centered Modal</h4>
        <p>
          Cras mattis consectetur purus sit amet fermentum. Cras justo odio,
          dapibus ac facilisis in, egestas eget quam. Morbi leo risus, porta ac
          consectetur ac, vestibulum at eros.
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}
function MyVerticallyCenteredModal4(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Modal heading
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Centered Modal</h4>
        <p>data on Model 4</p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}
function MyVerticallyCenteredModal5(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Modal heading
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Centered Modal</h4>
        <p>data on Model 5</p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}
const SelectModels = (prop) => {
  const [modalShow1, setModalShow1] = React.useState(false);
  const [modalShow2, setModalShow2] = React.useState(false);
  const [modalShow3, setModalShow3] = React.useState(false);
  const [modalShow4, setModalShow4] = React.useState(false);
  const [modalShow5, setModalShow5] = React.useState(false);
  const [selectedModels, setSelectedModels] = useState([]);

  const classesVal = prop.noOfClasses;

  const handleModelChange = (event) => {
    const value = event.target.value;
    if (event.target.checked) {
      setSelectedModels([...selectedModels, value]); // Add selected model
    } else {
      setSelectedModels(selectedModels.filter((model) => model !== value)); // Remove unselected model
    }
    console.log(selectedModels);
  };

  const handleModel = async () => {
    const email = "abcd@gmail.com";
    console.log(classesVal);

    try {
      const response = await axios({
        method: "post",
        url: `http://localhost:8000/run_model/${email}`,
        data: {
          email: email,
          no_of_classes: classesVal,
          selectedModels: selectedModels, // Send selected models to the backend
        },
      });
    } catch (error) {
      console.error(error);
    }
  };
  return (
    <div className="flex flex-col items-center justify-center mt-12 gap-3 bg-green-300 border-black rounded-full ">
      <div className="text-black text-3xl underline mb-9 mt-3 ">
        Select Models
      </div>
      <div className="text-black text-lg flex justify-center underline mb-10 ">
        Click on the Corresponding Model Button to know about its specifications
      </div>
      <div className="flex flex-row items-center justify-center text-2xl">
        <input
          type="checkbox"
          value="GoogleNet"
          className="size-5 "
          checked={selectedModels.includes("GoogleNet")}
          onChange={handleModelChange}
        />

        <Button
          variant="primary"
          onClick={() => setModalShow1(true)}
          className="ml-7"
        >
          GoogleNet
        </Button>

        <MyVerticallyCenteredModal1
          show={modalShow1}
          onHide={() => setModalShow1(false)}
        />
      </div>
      <div className="flex flex-row items-center justify-center text-2xl">
        <input
          type="checkbox"
          value="AlexNet"
          className="size-5 "
          checked={selectedModels.includes("AlexNet")}
          onChange={handleModelChange}
        />
        <Button
          variant="primary"
          onClick={() => setModalShow2(true)}
          className="ml-7"
        >
          AlexNet
        </Button>

        <MyVerticallyCenteredModal2
          show={modalShow2}
          onHide={() => setModalShow2(false)}
        />
      </div>
      <div className="flex flex-row items-center justify-center text-2xl">
        <input
          type="checkbox"
          value="VGG16"
          className="size-5 "
          checked={selectedModels.includes("VGG16")}
          onChange={handleModelChange}
        />
        <Button
          variant="primary"
          onClick={() => setModalShow3(true)}
          className="ml-7"
        >
          VGG16
        </Button>

        <MyVerticallyCenteredModal3
          show={modalShow3}
          onHide={() => setModalShow3(false)}
        />
      </div>
      <div className="flex flex-row items-center justify-center text-2xl mb-4">
        <input
          type="checkbox"
          value="MobileNet"
          className="size-5 "
          checked={selectedModels.includes("MobileNet")}
          onChange={handleModelChange}
        />
        <Button
          variant="primary"
          onClick={() => setModalShow5(true)}
          className="ml-7"
        >
          MobileNetV2
        </Button>

        <MyVerticallyCenteredModal5
          show={modalShow5}
          onHide={() => setModalShow5(false)}
        />
      </div>

      <button
        className="w-28 hover:bg-red-600 rounded-md mb-10"
        onClick={handleModel}
      >
        Start Training
      </button>
    </div>
  );
};

export default SelectModels;
