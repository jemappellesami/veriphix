OPENQASM 2.0;
include "qelib1.inc";
qreg q122[5];
rx(5*pi/4) q122[4];
cx q122[3],q122[4];
cx q122[3],q122[2];
cx q122[2],q122[1];
cx q122[0],q122[1];
