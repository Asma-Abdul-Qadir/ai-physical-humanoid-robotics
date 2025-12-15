// This is the base NavbarItem component that handles different types of navbar items
// The custom-AuthButton type has been removed as requested

import React from 'react';
import DefaultNavbarItem from '@theme-original/NavbarItem';

const OriginalNavbarItem = (props) => {
  // For all types, use the default navbar item since auth button is removed
  return <DefaultNavbarItem {...props} />;
};

export default OriginalNavbarItem;