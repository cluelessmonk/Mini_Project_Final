import React, { useState } from "react";
import Button from "react-bootstrap/Button";
import Modal from "react-bootstrap/Modal";

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
const SelectModels = () => {
  // Define state to keep track of checkbox values
  const [modalShow1, setModalShow1] = React.useState(false);
  const [modalShow2, setModalShow2] = React.useState(false);
  const [modalShow3, setModalShow3] = React.useState(false);
  const [modalShow4, setModalShow4] = React.useState(false);

  const [checkboxValues, setCheckboxValues] = useState({
    Model1: false,
    Model2: false,
    Model3: false,
    Model4: false,
    // Add more options as needed
  });

  // Function to handle checkbox changes
  const handleCheckboxChange = async (event) => {
    const { name, checked } = event.target;
    setCheckboxValues({ ...checkboxValues, [name]: checked });
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
          name="Model1"
          className="size-5 "
          checked={checkboxValues.Model1}
          onChange={handleCheckboxChange}
        />

        <Button
          variant="primary"
          onClick={() => setModalShow1(true)}
          className="ml-7"
        >
          Model1
        </Button>

        <MyVerticallyCenteredModal1
          show={modalShow1}
          onHide={() => setModalShow1(false)}
        />
      </div>
      <div className="flex flex-row items-center justify-center text-2xl">
        <input
          type="checkbox"
          name="Model2"
          className="size-5 "
          checked={checkboxValues.Model2}
          onChange={handleCheckboxChange}
        />
        <Button
          variant="primary"
          onClick={() => setModalShow2(true)}
          className="ml-7"
        >
          Model1
        </Button>

        <MyVerticallyCenteredModal2
          show={modalShow2}
          onHide={() => setModalShow2(false)}
        />
      </div>
      <div className="flex flex-row items-center justify-center text-2xl">
        <input
          type="checkbox"
          name="Model3"
          className="size-5 "
          checked={checkboxValues.Model3}
          onChange={handleCheckboxChange}
        />
        <Button
          variant="primary"
          onClick={() => setModalShow3(true)}
          className="ml-7"
        >
          Model2
        </Button>

        <MyVerticallyCenteredModal3
          show={modalShow3}
          onHide={() => setModalShow3(false)}
        />
      </div>
      <div className="flex flex-row items-center justify-center text-2xl mb-4">
        <input
          type="checkbox"
          name="Model4"
          className="size-5 "
          checked={checkboxValues.Model4}
          onChange={handleCheckboxChange}
        />
        <Button
          variant="primary"
          onClick={() => setModalShow4(true)}
          className="ml-7"
        >
          Model4
        </Button>

        <MyVerticallyCenteredModal4
          show={modalShow4}
          onHide={() => setModalShow4(false)}
        />
      </div>

      <button className="w-28 hover:bg-red-600 rounded-md mb-10">
        Start Training
      </button>
    </div>
  );
};

export default SelectModels;
