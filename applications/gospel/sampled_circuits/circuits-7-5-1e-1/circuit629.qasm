OPENQASM 2.0;
include "qelib1.inc";
qreg q630[7];
cx q630[4],q630[5];
cx q630[3],q630[4];
cx q630[2],q630[3];
cx q630[2],q630[1];
cx q630[0],q630[1];
