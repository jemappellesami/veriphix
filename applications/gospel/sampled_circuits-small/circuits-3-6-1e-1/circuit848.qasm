OPENQASM 2.0;
include "qelib1.inc";
qreg q849[3];
rx(pi/4) q849[0];
rx(pi/2) q849[2];
cx q849[1],q849[2];
cx q849[1],q849[0];
rx(pi/4) q849[1];
